
import os
import tempfile
import json

import streamlit as st

from enhanced_shopping_agent import agent

from shopping_agent import (
    describe_product_image,
    search_products,
    get_ratings
)

from memory_store import get_orders, get_preferences
from langchain_core.messages import SystemMessage
from memory_tools import get_user_preferences


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None


st.title("🛒 AI Shopping Assistant")

st.caption(
    "Tell me what you want — I'll search, rate, and order the best match for you."
)


# ---------------------------------------------------------------------------
# Sidebar — Shop by Image
# ---------------------------------------------------------------------------

with st.sidebar:

    st.header("Shop by Image")

    st.caption(
        "Upload a photo of a product and I'll find similar items in our store."
    )

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            width="stretch"
        )

    if uploaded_file and st.button(
        "Find similar products",
        width="stretch"
    ):

        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(uploaded_file.getvalue())

            image_path = tmp.name

        image_prompt = (
            "I uploaded a product image. Please analyze it and "
            "find similar products in the store."
        )

        st.session_state.messages.append({
            "role": "user",
            "content": image_prompt,
            "filename": uploaded_file.name
        })

        st.session_state.pending_image = {
            "filename": uploaded_file.name,
            "image_path": image_path
        }

        st.rerun()


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if (
            msg["role"] == "user"
            and "filename" in msg
        ):

            st.markdown(
                f"Searching by image: **{msg['filename']}**"
            )

        else:

            st.markdown(
                msg["content"].replace("$", r"\$")
            )


# ---------------------------------------------------------------------------
# Run image search directly
# ---------------------------------------------------------------------------

if st.session_state.pending_image:

    image_info = st.session_state.pending_image

    image_path = image_info["image_path"]

    try:

        with st.chat_message("assistant"):

            # -------------------------------------------------------
            # STEP 1: Analyze image
            # -------------------------------------------------------

            with st.spinner(
                "Step 1/3: Identifying your product..."
            ):

                print("\n[IMAGE SEARCH] Starting image analysis")

                image_result = describe_product_image.invoke({
                    "image_path": image_path
                })

                print(
                    "[IMAGE SEARCH] Response:",
                    image_result
                )

                if image_result.startswith("ERROR:"):

                    raise Exception(image_result)

                image_data = json.loads(image_result)

                search_query = image_data.get("search_query")

                is_organic = image_data.get("is_organic")

                if not search_query:

                    raise Exception(
                        "Could not identify a searchable product."
                    )

                print(
                    "[IMAGE SEARCH] Identified:",
                    search_query
                )


            # -------------------------------------------------------
            # STEP 2: Retrieve preferences and search products
            # -------------------------------------------------------

            with st.spinner(
                "Step 2/3: Searching your store..."
            ):

                preferences = get_user_preferences.invoke({})

                if isinstance(preferences, str):

                    try:

                        preferences = json.loads(preferences)

                    except json.JSONDecodeError:

                        preferences = {}

                if not isinstance(preferences, dict):

                    preferences = {}

                search_args = {
                    "query": search_query,
                    "is_organic": is_organic
                }

                max_price = preferences.get("max_price")

                if max_price is not None:

                    search_args["max_price"] = float(max_price)

                products_result = search_products.invoke(
                    search_args
                )

                products = json.loads(products_result)

                print(
                    "[PRODUCT SEARCH] Found:",
                    len(products)
                )

            # -------------------------------------------------------
            # STEP 3: Retrieve ratings in one batch
            # -------------------------------------------------------

            with st.spinner("Step 3/3: Checking product ratings..."):

                rated_products = products[:5]

                try:
                    product_ids = [
                        product["id"] for product in rated_products
                    ]

                    rating_result = get_ratings.invoke({
                        "product_ids": product_ids
                    })

                    rating_data = json.loads(rating_result)

                    ratings_map = {
                        item["product_id"]: item
                        for item in rating_data
                    }

                    for product in rated_products:
                        rating = ratings_map.get(product["id"], {})

                        product["rating"] = rating.get("average_rating")
                        product["review_count"] = rating.get("review_count", 0)

                except Exception as rating_error:
                    print("[RATING ERROR]", repr(rating_error))

                    for product in rated_products:
                        product["rating"] = None
                        product["review_count"] = 0


            # -------------------------------------------------------
            # Display image-search results
            # -------------------------------------------------------

            if not rated_products:

                response = (
                    f"I identified the product as {search_query}, "
                    "but couldn't find matching products in your store."
                )

            else:

                response = (
                    f"Identified product: "
                    f"{image_data.get('product_type', search_query)}\n\n"
                    "Here are the matching products:\n\n"
                )

                for index, product in enumerate(
                    rated_products,
                    start=1
                ):

                    rating = product.get("rating")

                    rating_text = (
                        f"{rating} stars"
                        if rating is not None
                        else "Rating unavailable"
                    )

                    organic_text = (
                        "Organic"
                        if product["is_organic"]
                        else "Non-organic"
                    )

                    response += (
                        f"#{index}. {product['name']} "
                        f"(ID:{product['id']}) — "
                        f"${product['price']:.2f} — "
                        f"★{rating_text} — {organic_text}\n\n"
                    )

                response += (
                    "Would you like to order any of these products?"
                )

            st.markdown(
                response.replace("$", r"\$")
            )


        # Save assistant response in chat history

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })


    except Exception as e:

        print(
            "[IMAGE SEARCH ERROR]",
            repr(e)
        )

        error_response = f"Image search failed: {str(e)}"

        st.error(error_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_response
        })


    finally:

        if os.path.exists(image_path):

            os.remove(image_path)

        st.session_state.pending_image = None

    st.rerun()


# ---------------------------------------------------------------------------
# Text input — Existing shopping functionality preserved
# ---------------------------------------------------------------------------

if prompt := st.chat_input(
    "e.g. I want organic honey under $15 with 4+ rating"
):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

        # -----------------------------------------------------------
        # Handle order history directly from SQLite
        # -----------------------------------------------------------

        if any(
            phrase in prompt.lower()
            for phrase in [
                "what have i ordered",
                "order history",
                "previous orders",
                "orders before",
                "my past orders"
            ]
        ):

            orders = get_orders()

            if not orders:

                response = (
                    "You have no previous orders recorded."
                )

            else:

                response = (
                    "Here is your actual order history:\n\n"
                )

                for order in orders:

                    response += (
                        f"- Product: {order['product_name']} | "
                        f"Price: ${order['price']:.2f}\n"
                    )


        # -----------------------------------------------------------
        # Normal shopping through enhanced agent
        # -----------------------------------------------------------

        else:

            with st.spinner("Thinking..."):

                try:
                    recent_messages = st.session_state.messages[-5:]

                    # Ensure history starts with a user message
                    if recent_messages and recent_messages[0].get("role") == "assistant":
                        recent_messages = recent_messages[1:]

                    # Retrieve preferences directly from SQLite
                    preferences = get_preferences()

                    preference_context = SystemMessage(
                        content=(
                            "Saved shopping preferences:\n"
                            f"Organic preference: {preferences.get('prefers_organic')}\n"
                            f"Maximum price: {preferences.get('max_price')}\n"
                            "Apply these preferences to product searches unless "
                            "the user explicitly overrides them for the current search."
                        )
                    )

                    agent_messages = [preference_context] + recent_messages

                    print("Total conversation messages:", len(st.session_state.messages))
                    print("Messages sent to agent:", len(recent_messages))
                    print("Saved preferences retrieved:", preferences)

                    result = agent.invoke(
                        {"messages": recent_messages},
                        config={
                            "recursion_limit": 12,
                            "run_name": "Shopping_Agent_Trace"
                        }
                    )
                    response = result["messages"][-1].content

                    if not isinstance(response, str):

                        response = str(response)

                    response = response.replace("`", "")

                except Exception as e:

                    response = (
                        f"Sorry, something went wrong: {str(e)}"
                    )


        st.markdown(
            response.replace("$", r"\$")
        )


    # Save assistant response

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()
