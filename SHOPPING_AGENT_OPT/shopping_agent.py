
import base64
import json
import os
import re
import sqlite3
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_ratings_for_products


os.environ["LANGCHAIN_PROJECT"] = "Shopping_agent"

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

print("SHOPPING DATABASE:", DB_PATH)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_effort="low",
    include_reasoning=False,
    timeout=30,
    max_retries=1
)

vision_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    timeout=45,
    max_retries=1
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None
) -> str:
    """Search products by matching product names first, then descriptions, with optional filters."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        stop_words = {
            "a", "an", "the", "and", "or",
            "of", "for", "with", "in", "on",
            "i", "want", "need", "looking", "buy",
            "under", "below", "less", "than",
            "price", "product", "products",
            "raw", "fresh", "organic",
            "nut", "nuts"
        }

        keywords = [
            word.strip().lower()
            for word in query.split()
            if len(word.strip()) > 2
            and word.strip().lower() not in stop_words
        ]

        # Normalize plural words (existing logic retained).
        keywords = [
            word[:-1] if word.endswith("s") and len(word) > 3 else word
            for word in keywords
        ]

        keywords = list(dict.fromkeys(keywords))

        print("[PRODUCT SEARCH] Keywords:", keywords)

        if not keywords:
            return json.dumps([])

        # Apply optional filters.
        filter_sql = ""
        filter_params = []

        if max_price is not None:
            filter_sql += " AND price <= ?"
            filter_params.append(max_price)

        if is_organic is not None:
            filter_sql += " AND is_organic = ?"
            filter_params.append(1 if is_organic else 0)

        # STEP 1: Search product names.
        name_conditions = [
            "LOWER(name) LIKE ?" for word in keywords
        ]

        name_params = [
            f"%{word}%" for word in keywords
        ]

        name_sql = f"""
            SELECT
                id,
                name,
                category,
                price,
                is_organic
            FROM products
            WHERE {" AND ".join(name_conditions)}
            {filter_sql}
            ORDER BY name
            LIMIT 10
        """

        name_params.extend(filter_params)

        cursor.execute(name_sql, name_params)
        rows = cursor.fetchall()

        print(
            "[PRODUCT SEARCH] Product-name matches:",
            len(rows)
        )

        # STEP 2: Search descriptions only if no product name matches.
        if not rows:

            description_conditions = [
                "LOWER(description) LIKE ?" for word in keywords
            ]

            description_params = [
                f"%{word}%" for word in keywords
            ]

            description_sql = f"""
                SELECT
                    id,
                    name,
                    category,
                    price,
                    is_organic
                FROM products
                WHERE {" AND ".join(description_conditions)}
                {filter_sql}
                ORDER BY name
                LIMIT 10
            """

            description_params.extend(filter_params)

            cursor.execute(
                description_sql,
                description_params
            )

            rows = cursor.fetchall()

            print(
                "[PRODUCT SEARCH] Description matches:",
                len(rows)
            )

        # Convert database rows into JSON.
        # Both queries return exactly five columns.
        products = [
            {
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "price": row[3],
                "is_organic": bool(row[4]),
            }
            for row in rows
        ]

        print(
            "[PRODUCT SEARCH] Final matching products:",
            len(products)
        )

        print(
            "[PRODUCT SEARCH] Results:",
            [product["name"] for product in products]
        )

        return json.dumps(products)

    except Exception as e:
        print("[PRODUCT SEARCH ERROR]", repr(e))
        return json.dumps({"error": str(e)})

    finally:
        conn.close()

@tool
def get_ratings(product_ids: list[int]) -> str:
    """
    Get average customer ratings and review counts
    for multiple products in one database query.
    """

    results = get_ratings_for_products(product_ids)

    return json.dumps(results)

@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID.
    Saves the order to the database and returns confirmation.
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        print("[CHECKOUT] Requested product ID:", product_id)
        cursor.execute(
            "SELECT name, price FROM products WHERE id = ?",
            (product_id,)
        )

        row = cursor.fetchone()
        print("[CHECKOUT] Product found:", row)

        if not row:
            return f"Error: product with ID {product_id} not found."

        name, price = row

        cursor.execute(
            """
            INSERT INTO orders (product_id, product_name, price)
            VALUES (?, ?, ?)
            """,
            (product_id, name, price)
        )
        print("[CHECKOUT] Inserted order ID:", cursor.lastrowid)

        order_id = cursor.lastrowid

        conn.commit()
        print("[CHECKOUT] Transaction committed successfully")

    finally:
        conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been "
        f"successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. "
        f"Thank you for shopping with us!"
    )


# ---------------------------------------------------------------------------
# Image Analysis Tool
# ---------------------------------------------------------------------------

@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze an uploaded product image and return product attributes
    as valid JSON.
    """

    print("\n[IMAGE SEARCH] Starting image analysis...")
    print("[IMAGE SEARCH] Image path:", image_path)

    try:

        # Check whether the image exists
        if not os.path.exists(image_path):
            return f"ERROR: Image file not found: {image_path}"

        file_size = os.path.getsize(image_path)

        print(
            "[IMAGE SEARCH] File size:",
            file_size,
            "bytes"
        )

        if file_size == 0:
            return "ERROR: Uploaded image is empty."

        if file_size > 20 * 1024 * 1024:
            return "ERROR: Image exceeds Groq's 20 MB image limit."

        # Convert image into Base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(
                f.read()
            ).decode("utf-8")

        # Identify image format
        ext = os.path.splitext(image_path)[1].lower()

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }

        mime = mime_types.get(ext)

        if not mime:
            return f"ERROR: Unsupported image format: {ext}"

        print("[IMAGE SEARCH] Image encoded successfully.")

        # Prepare vision model message
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{image_data}"
                    },
                },
                {
                    "type": "text",
                    "text": """
                    Identify the main physical product in this image.

                    Return ONLY a valid JSON object.

                    Do not include Markdown code fences.
                    Do not include explanations outside the JSON.

                    Use exactly these field names:

                    {
                        "product_type": "product name",
                        "search_query": "short searchable keyword",
                        "is_organic": null,
                        "description": "visible product description"
                    }

                    Rules:

                    1. product_type should identify the product.
                    2. search_query should contain short, useful
                       product keywords for database searching.
                    3. is_organic must be true, false, or null.
                    4. Do not guess whether a product is organic.
                    5. Identify only what is actually visible.
                    """
                },
            ]
        )

        # Call vision model
        print("[IMAGE SEARCH] Sending image to Groq vision model...")

        response = vision_llm.invoke([message])

        print("[IMAGE SEARCH] Vision response received.")

        result = response.content

        if not isinstance(result, str):
            result = str(result)

        print(
            "[IMAGE SEARCH] Raw vision response:",
            repr(result)
        )

        # -----------------------------------------------------------
        # Extract JSON from model response
        # -----------------------------------------------------------

        cleaned_result = result.strip()

        # Remove opening Markdown code fence
        cleaned_result = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned_result,
            flags=re.IGNORECASE
        )

        # Remove closing Markdown code fence
        cleaned_result = re.sub(
            r"\s*```$",
            "",
            cleaned_result
        )

        cleaned_result = cleaned_result.strip()

        # Extract JSON object if extra text is present
        json_match = re.search(
            r"\{.*\}",
            cleaned_result,
            re.DOTALL
        )

        if not json_match:
            print("[IMAGE SEARCH ERROR] No JSON object found.")

            return (
                "ERROR: Vision model did not return "
                "a valid JSON object."
            )

        json_text = json_match.group(0)

        # Parse JSON
        try:
            image_details = json.loads(json_text)

        except json.JSONDecodeError as e:
            print(
                "[IMAGE SEARCH ERROR] JSON parsing failed:",
                repr(e)
            )

            return (
                f"ERROR: Invalid JSON returned by vision model: {str(e)}"
            )

        if not isinstance(image_details, dict):
            return "ERROR: Vision model response must be a JSON object."

        # -----------------------------------------------------------
        # Normalize model field names
        # -----------------------------------------------------------

        if "search query" in image_details:
            image_details["search_query"] = image_details.pop(
                "search query"
            )

        if "search-query" in image_details:
            image_details["search_query"] = image_details.pop(
                "search-query"
            )

        # Validate search query
        search_query = image_details.get("search_query")

        if not isinstance(search_query, str) or not search_query.strip():
            return (
                "ERROR: Vision model did not return "
                "a valid search_query."
            )

        # Normalize organic value
        organic_value = image_details.get("is_organic")

        if organic_value not in (True, False, None):
            image_details["is_organic"] = None

        # Print extracted information
        print(
            "[IMAGE SEARCH] Identified product:",
            image_details.get("product_type")
        )

        print(
            "[IMAGE SEARCH] Search query:",
            image_details["search_query"]
        )

        print(
            "[IMAGE SEARCH] Organic:",
            image_details.get("is_organic")
        )

        # Return clean JSON to the application
        final_result = json.dumps(image_details)

        print("[IMAGE SEARCH] Final JSON:", final_result)

        return final_result

    except Exception as e:
        print("[IMAGE SEARCH ERROR]", repr(e))

        return f"ERROR: Image analysis failed: {str(e)}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    tools=[
        search_products,
        get_ratings,
        checkout,
        describe_product_image
    ],

    model=llm,

    system_prompt=(

        "You are a helpful shopping assistant. "
        "Follow these rules strictly.\n\n"

        "IMAGE SEARCH — when the user provides an image path:\n"

        "1. Call describe_product_image with the path to identify the product.\n"

        "2. Use the returned search_query and is_organic "
        "to call search_products.\n"

        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"

        "BROWSING — when the user describes what they want to buy:\n"

        "1. Call search_products to find matching items "
        "(apply any price/organic filters given).\n"

        "2. Call get_ratings once with all candidate product IDs.\n"

        "3. Use the returned ratings to filter the products.\n"

        "4. Present qualifying products as a numbered list. "
        "For each item use this exact format "
        "(plain text, no backticks, no code blocks, no bold, no italic):\n\n"

        "   #<number>. <name> (ID:<product_id>) — "
        "$<price> ★<rating> — <organic or non-organic>\n\n"

        "   Add a blank line between each product entry for readability. "
        "Always include (ID:X) so you can reference it later.\n"

        "5. If only one product qualifies, still show it in the list "
        "and ask: 'Would you like to order it? "
        "Just say yes or give me the number.'\n"

        "6. Do NOT call checkout at this stage.\n\n"

        "ORDERING — when the user confirms they want to buy "
        "(e.g. 'yes', 'sure', 'go ahead', 'order number 2', "
        "'the first one', 'get me #3'):\n"

        "1. Look at your previous message to find the (ID:X) "
        "for the chosen product "
        "(if only one was listed and the user says 'yes', "
        "use that product's ID).\n"

        "2. Call checkout with that product_id "
        "(the number from (ID:X)).\n"

        "3. Confirm the order to the user in plain text.\n\n"

        "Never place an order unless the user explicitly confirms. "

        "Never guess a product_id — always take it from the "
        "(ID:X) in your own previous message."
    ),
)


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey "
                        "with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )

    print(result["messages"][-1].content)
