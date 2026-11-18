import os
from langsmith import evaluate
from playwright.sync_api import sync_playwright

DATASET_NAME = "Shopping_Agent_Test_Dataset"

RAILWAY_URL = os.environ["RAILWAY_APP_URL"]


def run_live_agent(inputs: dict) -> dict:
    question = inputs["question"]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(RAILWAY_URL, wait_until="domcontentloaded")

            chat_input = page.locator(
                "textarea[data-testid='stChatInputTextArea']"
            )
            chat_input.wait_for(timeout=60000)

            chat_input.fill(question)
            chat_input.press("Enter")

            # Wait until the chatbot displays its answer.
            page.wait_for_function(
                """(question) => {
                    const messages = Array.from(
                        document.querySelectorAll(
                            '[data-testid="stChatMessage"]'
                        )
                    );

                    if (messages.length < 2) return false;

                    const lastMessage =
                        messages[messages.length - 1].innerText.trim();

                    return lastMessage.length > 0 &&
                           lastMessage !== question;
                }""",
                arg=question,
                timeout=120000
            )

            messages = page.locator(
                "[data-testid='stChatMessage']"
            )

            answer = messages.last.locator("[data-testid='stMarkdownContainer']").inner_text()

            return {"answer": answer}

        finally:
            browser.close()


if __name__ == "__main__":

    results = evaluate(
        run_live_agent,
        data=DATASET_NAME,
        experiment_prefix="shopping-agent-railway",
        description="Evaluation of live Railway Shopping Agent",
        max_concurrency=1,
    )

    print(results)
