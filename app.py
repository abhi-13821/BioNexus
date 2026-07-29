from frontend import literature_search
from frontend import drug_discovery
from frontend import smiles_analyzer
from frontend import drug_info
from frontend import ai_assistant
from frontend import knowledge_graphs
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="BioNexus",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🧬 BioNexus")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📄 Literature Search",
        "💊 Drug Information",
        "💊 Drug Discovery",
        "🧪 SMILES Analyzer",
        "🕸️ Knowledge Graph",
        "🤖 AI Assistant",
        "📊 Evaluation",
        "⚙️ Settings"
    ]
)

# Home Page
if page == "🏠 Home":

    st.title("🧬 BioNexus")

    st.subheader(
        "AI-Powered Biomedical Drug Discovery Assistant"
    )

    st.success(
        "System Initialized Successfully ✅"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🧬 Biomedical Literature Search")
        st.info("💊 Drug Discovery")
        st.info("🧪 SMILES Analyzer")
        st.info("🕸️ Knowledge Graph")

    with col2:
        st.info("🤖 AI Assistant")
        st.info("🧠 Fine-tuned LLM")
        st.info("📈 Custom Embeddings")
        st.info("📊 Evaluation Dashboard")

    st.markdown("---")

    st.write(
        "Welcome to BioNexus Version 1.0 🚀"
    )


# Other pages (temporary placeholders)
elif page == "📄 Literature Search":
    literature_search.show()

elif page == "💊 Drug Information":
    drug_info.show()

elif page == "💊 Drug Discovery":
    drug_discovery.main()

elif page == "🧪 SMILES Analyzer":
    smiles_analyzer.show()

elif page == "🕸️ Knowledge Graph":
    knowledge_graphs.show()

elif page == "🤖 AI Assistant":
    ai_assistant.main()

elif page == "📊 Evaluation":
    st.title("📊 Evaluation")
    st.write("Module under development...")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.write("Module under development...")
