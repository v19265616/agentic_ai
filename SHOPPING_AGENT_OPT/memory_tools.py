import json
from langchain.tools import tool
from memory_store import get_orders, get_preferences, save_preferences


@tool
def get_order_history(limit: int = 20) -> str:
    """Get the user's previously placed orders from the existing SQLite orders table."""
    return json.dumps(get_orders(limit), indent=2)


@tool
def get_user_preferences() -> str:
    """Retrieve saved shopping preferences before searching for products."""
    return json.dumps(get_preferences(), indent=2)


@tool
def save_user_preferences(prefers_organic: bool = None, max_price: float = None) -> str:
    """Save preferences only when the user explicitly asks to remember them.
    Example: prefers_organic=true and max_price=20. Omit unchanged fields."""
    if max_price is not None and max_price <= 0:
        return "Maximum price must be greater than zero."
    return json.dumps(
        save_preferences(prefers_organic=prefers_organic, max_price=max_price),
        indent=2,
    )

