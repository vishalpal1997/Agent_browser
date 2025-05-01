import asyncio
import os
from dotenv import load_dotenv
import gradio as gr
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
import logging
from io import StringIO

# Setup logging
log_stream = StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(log_stream), logging.StreamHandler()]
)

# Load environment
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=SecretStr(api_key),
    temperature=0
)

# Store conversation history
conversation_history = []

# Create and set a persistent event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

async def async_run_task(task):
    """Execute task asynchronously and return output"""
    global conversation_history
    
    # Clear previous logs
    log_stream.truncate(0)
    log_stream.seek(0)
    
    try:
        agent = Agent(task=task, llm=llm)
        result = await agent.run()
        
        # Get complete logs
        full_logs = log_stream.getvalue()
        
        # Store conversation context
        conversation_history.append({
            'task': task,
            'logs': full_logs,
            'result': str(result) if result else None
        })
        
        return f"## Execution Logs\n```\n{full_logs}\n```\n\n## Result\n{result}"
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nLogs:\n{log_stream.getvalue()}"
        conversation_history.append({'error': error_msg})
        return error_msg

def run_task(task):
    """Wrapper to run async task from sync context"""
    return loop.run_until_complete(async_run_task(task))

async def async_ask_question(question):
    """Answer questions about the execution asynchronously"""
    if not conversation_history:
        return "Please execute a task first."
    
    last_execution = conversation_history[-1]
    context = f"""
    Last Task: {last_execution.get('task', 'N/A')}
    Execution Logs: {last_execution.get('logs', 'No logs available')}
    Raw Result: {last_execution.get('result', 'No explicit result')}
    """
    
    try:
        response = await llm.ainvoke(
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the execution logs above:"
        )
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def ask_question(question):
    """Wrapper to run async question answering from sync context"""
    return loop.run_until_complete(async_ask_question(question))

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# Browser Automation Agent")
    
    with gr.Row():
        with gr.Column():
            task_input = gr.Textbox(label="Enter Task", placeholder="Search for...")
            run_btn = gr.Button("Execute")
            output_area = gr.Textbox(label="Execution Output", lines=10, interactive=False)
        
        with gr.Column():
            question_input = gr.Textbox(label="Ask about the execution")
            ask_btn = gr.Button("Ask Question")
            answer_area = gr.Textbox(label="Answer", interactive=False)

    run_btn.click(
        fn=run_task,
        inputs=task_input,
        outputs=output_area
    )
    
    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_area
    )

if __name__ == "__main__":
    try:
        app.launch()
    finally:
        # Clean up the event loop when done
        loop.close()
