import streamlit as st
from agents.router import MathRoutingAgent
import json
from datetime import datetime
import os
import traceback

# Page config
st.set_page_config(
    page_title="Math Routing Agent",
    page_icon="🧮",
    layout="wide"
)

# Initialize session state with error handling
@st.cache_resource
def initialize_router():
    """Initialize the router with caching"""
    try:
        return MathRoutingAgent()
    except Exception as e:
        st.error(f"Failed to initialize router: {e}")
        st.error("Please check your configuration and API keys.")
        return None

if 'router' not in st.session_state:
    st.session_state.router = initialize_router()

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Check if router initialized successfully
if st.session_state.router is None:
    st.error("⚠️ Application failed to initialize. Please check your configuration.")
    st.stop()

# Title and description
st.title("🧮 Math Professor AI Agent")
st.markdown("### Ask me any mathematics question and get step-by-step solutions!")

# Sidebar for settings and info
with st.sidebar:
    st.header("📊 Agent Status")
    
    # Check component status
    try:
        if hasattr(st.session_state.router, 'guardrails'):
            st.success("✅ Guardrails Active")
        else:
            st.warning("⚠️ Guardrails Not Available")
            
        if hasattr(st.session_state.router, 'knowledge_base'):
            st.success("✅ Knowledge Base Ready")
        else:
            st.warning("⚠️ Knowledge Base Not Available")
            
        if hasattr(st.session_state.router, 'web_search'):
            st.success("✅ Web Search Available")
        else:
            st.warning("⚠️ Web Search Not Available")
            
        st.success("✅ Feedback System Online")
        
    except Exception as e:
        st.error(f"Status check failed: {e}")
    
    st.header("📝 Sample Questions")
    st.markdown("""
    - Solve x² + 5x + 6 = 0
    - Find derivative of 3x² + 2x + 1
    - Calculate area of circle with radius 5
    - Integrate ∫x²dx
    - Solve system: 2x + y = 5, x - y = 1
    """)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    # Query input
    user_query = st.text_area(
        "Enter your math question:",
        height=100,
        placeholder="e.g., Solve the equation 2x + 3 = 11"
    )
    
    if st.button("🔍 Get Solution", type="primary"):
        if user_query.strip():
            with st.spinner("Processing your question..."):
                try:
                    result = st.session_state.router.process_query(user_query)
                    
                    if result["success"]:
                        st.success("Solution Generated!")
                        
                        # Display source info
                        source_emoji = {
                            "knowledge_base": "📚",
                            "web_search_perplexity": "🧠",
                            "web_search_tavily": "🔍",
                            "web_search_duckduckgo": "🦆",
                            "ai_generated": "🤖"
                        }
                        
                        source_name = result['source']
                        emoji = source_emoji.get(source_name, '📝')
                        display_name = source_name.replace('_', ' ').title()
                        
                        st.info(f"{emoji} Source: {display_name}")
                        
                        # Display confidence if available
                        if 'confidence' in result and result['confidence'] > 0:
                            confidence_pct = int(result['confidence'] * 100)
                            st.info(f"🎯 Confidence: {confidence_pct}%")
                        
                        # Display solution
                        st.markdown("### 📝 Step-by-Step Solution")
                        st.markdown(result['solution'])
                        
                        # Store in session for feedback
                        st.session_state.current_query = user_query
                        st.session_state.current_solution = result['solution']
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            "timestamp": datetime.now(),
                            "query": user_query,
                            "solution": result['solution'],
                            "source": result['source']
                        })
                        
                    else:
                        st.error(f"❌ {result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    with st.expander("Debug Information (Click to expand)"):
                        st.code(traceback.format_exc())
        else:
            st.warning("Please enter a math question!")

with col2:
    # Feedback section
    if hasattr(st.session_state, 'current_solution'):
        st.header("💬 Feedback")
        
        rating = st.selectbox(
            "Rate the solution:",
            [1, 2, 3, 4, 5],
            index=4,
            format_func=lambda x: "⭐" * x
        )
        
        feedback_text = st.text_area(
            "Your feedback:",
            placeholder="How can we improve this solution?"
        )
        
        if st.button("Submit Feedback"):
            if feedback_text.strip():
                try:
                    feedback_result = st.session_state.router.process_feedback(
                        st.session_state.current_query,
                        st.session_state.current_solution,
                        feedback_text,
                        rating
                    )
                    
                    if feedback_result["success"]:
                        st.success("Thanks for your feedback!")
                        if rating < 3 and "refined_solution" in feedback_result:
                            st.markdown("### 🔄 Improved Solution")
                            st.markdown(feedback_result["refined_solution"])
                    else:
                        st.error(f"Feedback error: {feedback_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Failed to process feedback: {str(e)}")
            else:
                st.warning("Please provide feedback text!")

# Conversation history
if st.session_state.conversation_history:
    st.header("📚 Conversation History")
    
    for i, entry in enumerate(reversed(st.session_state.conversation_history[-5:])):
        with st.expander(f"Question {len(st.session_state.conversation_history)-i}: {entry['query'][:50]}..."):
            st.markdown(f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Source:** {entry['source'].replace('_', ' ').title()}")
            st.markdown("**Solution:**")
            st.markdown(entry['solution'])

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and HuggingFace")

# Debug panel (only show in development)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.header("🔧 Debug Information")
    st.sidebar.write("Router object:", type(st.session_state.router))
    st.sidebar.write("History length:", len(st.session_state.conversation_history))