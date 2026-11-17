
from dotenv import load_dotenv
from langsmith import evaluate

from enhanced_shopping_agent import agent
import time
load_dotenv()

DATASET_NAME = "Shopping_Agent_Test_Dataset"


def run_shopping_agent(inputs: dict) -> dict:
    time.sleep(30)
    question = inputs["question"]

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        },
        config={
            "recursion_limit": 12,
            "run_name": "Shopping_Agent_Evaluation"
        }
    )

    answer = result["messages"][-1].content

    return {"answer": answer}


if __name__ == "__main__":

    results = evaluate(
        run_shopping_agent,
        data=DATASET_NAME,
        experiment_prefix="shopping-agent-baseline",
        description="Baseline evaluation of Shopping Agent",
        max_concurrency=1,
    )

    print(results)
