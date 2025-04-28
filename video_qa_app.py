import streamlit as st
import streamlit.components.v1 as components
from retrieval_engine import RetrievalEngine

class VideoQAApp:
    
    def __init__(self):
        self.config = {
            'app': {
                'title': "🎬🔍 Multimodal RAG for Video Q&A",
                'description': "Ask any question about the video — the system finds and answers using the most relevant segments.",
                'layout': "wide",
                'video_url': "https://www.youtube.com/watch?v=dARr3lGKwk8"
            },
            'retrieval_methods': [
                "FAISS", 
                "pgvector-IVFFLAT", 
                "pgvector-HNSW", 
                "TF-IDF", 
                "BM25",
                "Hybrid"
            ]
        }
        self.engine = RetrievalEngine()
        self._setup_page_config()

    def _setup_page_config(self) -> None:
        st.set_page_config(
            page_title=self.config['app']['title'],
            layout=self.config['app']['layout']
        )


    def _render_header(self):
        st.markdown("""
        <div style='text-align: center; padding: 10px 0 30px 0;'>
            <h1 style='font-size: 3em; color: #4CAF50;'>🎬🔍 Multimodal RAG for Video Q&A</h1>
            <p style='font-size: 1.2em; color: #6c757d;'>Ask any question about the video and get answers with precise video segments.</p>
        </div>
        <hr style='margin-top: 0; margin-bottom: 2rem;'>
        """, unsafe_allow_html=True)

    def _render_sidebar(self) -> tuple:
        with st.sidebar:
            st.markdown("""
            <div class='sidebar-content'>
                <h2>🔧 Settings</h2>
                <hr>
            """, unsafe_allow_html=True)

            retrieval_method = st.selectbox(
                "🔎 Retrieval Method:",
                self.config['retrieval_methods']
            )
            top_k = st.slider(
                "🎯 Top-K Results:", 1, 5, 3
            )
            st.markdown("""
            <hr>
            <p style='font-size: 0.9em; color: grey;'>ℹ️ First search may be slower due to model loading.</p>
            </div>
            """, unsafe_allow_html=True)
        return retrieval_method, top_k

    def _render_video_player(self) -> None:
        with st.container():
            st.video(self.config['app']['video_url'])

    def _render_search_results(self, results: list) -> None:
        if not results:
            st.warning("⚠️ No results found. Try a different question.")
            return

        st.markdown("## 🔍 Search Results")
        
        for idx, res in enumerate(results):
            with st.expander(f"Result {idx+1}: {res['timestamp']:.2f}s"):
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.metric("⏱️ Timestamp", f"{res['timestamp']:.2f}s")
                    if 'score' in res:
                        st.metric("📈 Relevance", f"{res['score']:.3f}")

                with col2:
                    st.markdown("**📝 Transcript Segment:**")
                    st.write(res['text'])

                video_embed = f"""
                <div style='margin:10px 0;'>
                    <iframe width='100%' height='400' 
                        src='https://www.youtube.com/embed/dARr3lGKwk8?start={int(res['timestamp'])}&autoplay=0' 
                        frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' allowfullscreen>
                    </iframe>
                </div>
                """
                components.html(video_embed, height=420)
            st.markdown("---")

    def _execute_search(self, question: str, method: str, top_k: int) -> list:
        try:
            if method == "FAISS":
                return self.engine.semantic_search_faiss(question, top_k)
            elif method == "pgvector-IVFFLAT":
                return self.engine.semantic_search_pgvector(question, "ivfflat", top_k)
            elif method == "pgvector-HNSW":
                return self.engine.semantic_search_pgvector(question, "hnsw", top_k)
            elif method == "TF-IDF":
                return self.engine.lexical_search_tfidf(question, top_k)
            elif method == "BM25":
                return self.engine.lexical_search_bm25(question, top_k)
            elif method == "Hybrid":
                return self.engine.hybrid_search(question, top_k)
            else:
                st.error("🚨 Invalid retrieval method selected.")
                return []
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
            return []

    def run(self) -> None:
        self._render_header()

        retrieval_method, top_k = self._render_sidebar()
        self._render_video_player()

        st.markdown("## ❓ Ask a Question")

        with st.form("qa_form"):
            question = st.text_input(
                "", placeholder="e.g., What is the main topic of this video?", key="question_input"
            )
            submitted = st.form_submit_button("🚀 Search")

        if submitted:
            if not question.strip():
                st.warning("⚠️ Please enter a question first.")
            else:
                with st.spinner(f"Searching using {retrieval_method}..."):
                    results = self._execute_search(question, retrieval_method, top_k)
                    self._render_search_results(results)

if __name__ == "__main__":
    app = VideoQAApp()
    app.run()
