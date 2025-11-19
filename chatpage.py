import re
import gradio as gr
from gradio_modal import Modal
from conversationagent import ConversationAgent
from paperingestion import PaperVectorStore

# ====== Load user docstore ======
user_id = "demo_user"
user_store = PaperVectorStore(user_id)
user_vectorstore = user_store.vectorstore
agent = ConversationAgent(user_id=user_id, docstore=user_vectorstore)

# ====== Paper utilities ======
def get_ingested_papers():
    papers = []
    for paper_id, metadata in user_store.paper_metadata.items():
        papers.append({
            'id': paper_id,
            'Title': metadata.get('Title', 'Unknown'),
            'Authors': metadata.get('Authors', ''),
            'Year': metadata.get('Published', 'Unknown'),
            'Summary': metadata.get('Summary', ''),
            'ingested_at': metadata.get('ingested_at', '')
        })
    papers.sort(key=lambda x: x.get('ingested_at', ''), reverse=True)
    return papers

def prepare_dataset_samples():
    """Prepare papers for gr.Dataset display"""
    papers = get_ingested_papers()
    samples = []
    paper_ids = []
    
    for p in papers:
        # Format: [display_text, paper_id]
        display = f"{p['Title'][:60]}... • {p['Year']}" if len(p['Title']) > 60 else f"{p['Title']} • {p['Year']}"
        samples.append([display])
        paper_ids.append(p['id'])
    
    return samples, paper_ids

def open_paper_detail_from_dataset(evt: gr.SelectData, paper_ids_list):
    """Handle paper selection from dataset"""
    selected_idx = evt.index
    if selected_idx >= len(paper_ids_list):
        return gr.update(visible=True), gr.update(visible=False), "", "", gr.update(), gr.update()
    
    paper_id = paper_ids_list[selected_idx]
    metadata = user_store.paper_metadata.get(paper_id, {})
    
    if not metadata:
        return gr.update(visible=True), gr.update(visible=False), "", "", gr.update(), gr.update()
    
    pdf_url = metadata.get('pdf_url', "Couldn't load paper")
    pdf_html = f'<iframe src="{pdf_url}" width="100%" height="800px"></iframe>' if pdf_url else "Couldn't load paper"
    
    return (
        paper_id,
        gr.update(visible=False),                   # main_chat visibility
        gr.update(visible=True),                    # paper_detail visibility
        f"## {metadata.get('Title', 'Unknown')}",   # paper_title
        pdf_html,                                   # paper_content
        gr.update(),                                # paper_chatbot
        metadata.get('notes', "")                   # paper_notes
    )

def back_to_main():
    return 'main_chat', gr.update(visible=True), gr.update(visible=False), "", "", gr.update(), gr.update()

def validate_arxiv_id(arxiv_id: str) -> bool:
    pattern = r'^(\d{4}\.\d{4,5}(v\d+)?|[a-z]+(-[a-z]+)*/\d{7}(v\d+)?)$'
    return bool(re.match(pattern, arxiv_id))

def parse_ids(text):
    if not text:
        return [], []
    raw = re.split(r"[,\n\s]+", text)
    all_entries = [entry.strip(" '\"") for entry in raw if entry.strip(" '\"")]
    valid_ids, invalid_entries = [], []
    for entry in all_entries:
        if validate_arxiv_id(entry):
            valid_ids.append(entry)
        else:
            invalid_entries.append(entry)
    return valid_ids, invalid_entries

def submit_papers(text_input):
    valid_ids, invalid_entries = parse_ids(text_input)
    if not valid_ids and not invalid_entries:
        return "Please enter at least one ArXiv ID", gr.update(value=""), gr.update(), gr.update()
    
    validation_message = ""
    if invalid_entries:
        if len(invalid_entries) == 1:
            validation_message = f"❌ Entry not matching arXiv ID format was ignored: {invalid_entries[0]}"
        else:
            validation_message = f"❌ {len(invalid_entries)} entries not matching arXiv ID format were ignored: {', '.join(invalid_entries[:5])}"
            if len(invalid_entries) > 5:
                validation_message += f" ... and {len(invalid_entries) - 5} more"
        validation_message += "\n\n"
    
    if not valid_ids:
        return validation_message + "No valid arXiv IDs to process.", gr.update(value=""), gr.update(), gr.update()
    
    unique_ids = list(set(valid_ids))
    result = user_store.ingest_papers(unique_ids)
    
    # Reload vectorstore
    global user_vectorstore, agent
    user_vectorstore = user_store.vectorstore
    agent.docstore = user_vectorstore
    
    clear_input = gr.update(value="")
    final_message = validation_message + result.get('message', "Ingestion completed.")
    
    # Update dataset
    new_samples, new_ids = prepare_dataset_samples()
    
    return final_message, clear_input, gr.update(samples=new_samples), new_ids

def save_paper_notes(paper_id, notes):
    user_store.save_notes(paper_id, notes)
    return "Notes saved!"
# ====== Interface ======
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    # State to track paper IDs (needed for dataset click handling)
    paper_ids_state = gr.State(value=[])
    current_selection = gr.State(value= 'main_chat')
    
    # ------------------- Main Chat -------------------
    with gr.Column(scale=2) as main_chat:
        gr.Markdown("## PaperPal\nYour AI Research Assistant")
        initial_msg = "Hello! I am PaperPal your chat buddy!\n\nHow can I help you?"
        chatbot = gr.Chatbot(value=[{"role": "assistant", "content": initial_msg}], type="messages")
        generalchat = gr.ChatInterface(agent.chat_gen, chatbot=chatbot).queue()

    # ------------------- Paper Detail View -------------------
    with gr.Column(visible=False) as paper_detail:
        with gr.Row():
            back_button = gr.Button("← Back to Chat")
            with gr.Column(scale=10, min_width=0):
                paper_title = gr.Markdown("## Paper Title")
        
        with gr.Tabs():
            with gr.TabItem("Read"):
                paper_content = gr.HTML()
            with gr.TabItem("Chat"):
                paper_chatbot = gr.Chatbot(type="messages")
                paper_msg = gr.Textbox(label="Chat about this paper...", placeholder="Ask questions...", lines=1)
            with gr.TabItem("Notes"):
                paper_notes = gr.Textbox(label="Your Notes", placeholder="Add notes...", lines=10)
                feedback_markdown_for_notes = gr.Markdown() 
                save_notes_btn = gr.Button("Save Notes")
        
            save_notes_btn.click(
            fn=save_paper_notes,
            inputs=[current_selection, paper_notes],  
            outputs=[feedback_markdown_for_notes]
            )

        back_button.click(
            fn=back_to_main,
            outputs=[current_selection, main_chat, paper_detail, paper_title, paper_content, paper_chatbot, paper_notes]
        )

    # ------------------- Sidebar -------------------
    with gr.Sidebar():
        gr.Markdown("## Research Papers")
        search_box = gr.Textbox(label="Search papers", placeholder="Search...")
        
        # Use Dataset for clickable paper list
        papers_dataset = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=[["Loading papers..."]],  # Placeholder
            label="Click a paper to view",
            samples_per_page=10
        )
        
        # Handle dataset clicks
        papers_dataset.select(
            fn=open_paper_detail_from_dataset,
            inputs=[paper_ids_state],
            outputs=[current_selection, main_chat, paper_detail, paper_title, paper_content, paper_chatbot, paper_notes]
        )

        add_papers_button = gr.Button("+ Add Papers")
        with Modal(visible=False) as add_paper_modal:
            gr.Markdown("### Add new papers")
            arxiv_ids_input = gr.Textbox(
                label="Enter ArXiv IDs",
                placeholder="One per line or comma-separated",
                lines=3
            )
            feedback_markdown = gr.Markdown("")
            submit_btn = gr.Button("Submit")
            
            submit_btn.click(
                fn=submit_papers,
                inputs=[arxiv_ids_input],
                outputs=[feedback_markdown, arxiv_ids_input, papers_dataset, paper_ids_state]
            )
            
            add_papers_button.click(
                fn=lambda: (gr.update(visible=True), "", ""),
                outputs=[add_paper_modal, arxiv_ids_input, feedback_markdown]
            )
    
    # ------------------- Load papers on startup -------------------
    def load_papers_on_start():
        """Load papers when the app starts/refreshes"""
        samples, ids = prepare_dataset_samples()
        return gr.update(samples=samples if samples else [["No papers yet"]]), ids
    
    # Trigger on demo load
    demo.load(
        fn=load_papers_on_start,
        outputs=[papers_dataset, paper_ids_state]
    )

demo.launch()