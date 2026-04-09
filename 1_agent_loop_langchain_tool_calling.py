from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL="gpt-4o-mini"


@tool
def get_product_price(product: str) -> str:
    """Look up the price of a product in the catalog."""
    print(f"Looking up price for {product}...")
    prices = {"laptop": "999", "headphones": "499", "keyboard": "199"}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float: 
    """Apply a discount tier to a price and return the discounted price.
    Available tiers: bronze, silver, gold
    """
    print(f"Applying {discount_tier} discount to price {price}...")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


@traceable
def run_agent(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"openai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Running agent with question: {question}")
    print("=" * 50)

    messages = [
        SystemMessage(content="You are a helpful assistant. You have access to the product catalog tool and a discount tool.\n\n" \
        "STRICT RULES: you must follow these exactly:\n" \
            "1. NEVER assume or guess any product price\n" \
            "2. ALWAYS use the tools to get the price\n" \
            "3. NEVER apply a discount without a valid price\n" \
            "4. ALWAYS return the final price after discount\n" \
            "5. NEVER calculate discounts yourself, ALWAYS use the discount tool\n" \
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"Tool call: {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    
    print("Max iterations reached without a final answer.")
    return None
       


if __name__ == "__main__":
    print("Hello Langchain (.bind tools)")
    print()
    result = run_agent("What is the price of a laptop and what would it be with a silver discount?")