from dotenv import load_dotenv
import json

load_dotenv()

from openai import OpenAI
from langsmith import traceable

client = OpenAI()

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"


@traceable(run_type="tool")
def get_product_price(product: str):
    print(f"Looking up price for {product}...")
    prices = {"laptop": 999, "headphones": 499, "keyboard": 199}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str):
    print(f"Applying {discount_tier} discount to price {price}...")
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentage.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string"}
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply discount",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "discount_tier": {"type": "string"}
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]


@traceable(name="OpenAI Chat", run_type="llm")
def openai_chat_traced(messages):
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        tools=tools_for_llm,
        messages=messages
    )


@traceable(name="Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    messages = [
     {  
          "role": "system",
            "content": "You are a helpful assistant. You have access to the product catalog tool and a discount tool.\n\n" \
             "STRICT RULES: you must follow these exactly:\n" \
            "1. NEVER assume or guess any product price\n" \
            "2. ALWAYS use the tools to get the price\n" \
            "3. NEVER apply a discount without a valid price\n" \
            "4. ALWAYS return the final price after discount\n" \
            "5. NEVER calculate discounts yourself, ALWAYS use the discount tool\n" \
        },
        {"role": "user", "content": question}
    ]

    for _ in range(MAX_ITERATIONS):
        response = openai_chat_traced(messages)

        ai_message = response.choices[0].message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("Final Answer:", ai_message.content)
            return ai_message.content

        tool_call = tool_calls[0]

        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"Tool: {tool_name}, Args: {tool_args}")

        result = tools_dict[tool_name](**tool_args)

        print(f"Result: {result}")

        messages.append(ai_message)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    print("Max iterations reached")
    return None


if __name__ == "__main__":
    run_agent("What is the price of a laptop and apply silver discount?")