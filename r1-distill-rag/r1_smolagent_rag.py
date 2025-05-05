from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, tool, GradioUI, HfApiModel
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.style import Style
import time

load_dotenv()

# Setup rich console for pretty printing
console = Console()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id):
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        console.print(f"[bold green]🔧 Initializing HuggingFace model: {model_id}[/bold green]")
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        console.print(f"[bold blue]🔧 Initializing local Ollama model: {model_id}[/bold blue]")
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

# Create the reasoner with progress indication
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=True,
) as progress:
    task = progress.add_task("[cyan]Initializing reasoning model...", total=None)
    reasoning_model = get_model(reasoning_model_id)
    reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)
    progress.update(task, completed=1, description="[green]Reasoning model ready!")

# Initialize vector store and embeddings with visual feedback
console.print("[bold yellow]📚 Setting up vector database...[/bold yellow]")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)
console.print(f"[green]✅ Vector database loaded from {db_dir}[/green]")

@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response, so what you'll get back
    from this tool is a short answer to the user's question based on RAG context.

    Args:
        user_query: The user's question to query the vector database with.
    """
    console.print(Panel.fit(f"🔍 [bold]RAG Query:[/bold] {user_query}", 
                         style=Style(color="blue", bold=True)))
    
    # Search for relevant documents with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Searching vector database...", total=None)
        docs = vectordb.similarity_search(user_query, k=3)
        progress.update(task, completed=1, description="[green]Search complete!")
    
    # Show found documents count
    console.print(f"[dim]📄 Found {len(docs)} relevant documents[/dim]")
    
    # Combine document contents
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Create prompt with context
    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.
    
Context:
{context}

Question: {user_query}

Answer:"""
    
    # Get response from reasoning model with visual feedback
    console.print("[yellow]🤔 Reasoning about response...[/yellow]")
    start_time = time.time()
    response = reasoner.run(prompt, reset=False)
    elapsed = time.time() - start_time
    
    console.print(Panel.fit(
        Markdown(response),
        title="📝 RAG Response",
        subtitle=f"⏱️ {elapsed:.2f}s",
        border_style=Style(color="green")
    ))
    return response

# Create the primary agent with visual setup
console.print("[bold magenta]🚀 Initializing primary agent...[/bold magenta]")
tool_model = get_model(tool_model_id)
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model=tool_model, add_base_tools=False, max_steps=3)
console.print("[bold green]✅ Agent ready for interaction![/bold green]")

def main():
    console.print(Panel.fit(
        "[bold]SMOL Agents RAG System[/bold]\n"
        "Now launching Gradio interface...",
        style=Style(color="magenta", bold=True)
    ))
    GradioUI(primary_agent).launch()

if __name__ == "__main__":
    console.print("\n" * 2)  # Add some space
    console.rule("[bold blue]SMOL Agents RAG System[/bold blue]")
    main()