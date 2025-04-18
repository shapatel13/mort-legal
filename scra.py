import streamlit as st
import os
import json
from pathlib import Path
import datetime
import tempfile
import io
import PyPDF2
import re

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
import os


# Add these lines at the top of your script to disable tiktoken caching
os.environ["TIKTOKEN_CACHE_DIR"] = ""
# Alternatively, set it to a directory you have permission to write to:
# os.makedirs("./tiktoken_cache", exist_ok=True)
# os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cach
from llama_index.core import settings
# Create a cache directory in a location you have permissions for
os.makedirs("./cache_data", exist_ok=True)
settings.cache_dir = "./cache_data"  # <-- Corrected line

# Add Agno imports
from agno.agent import Agent
from agno.models.google import Gemini
from agno.knowledge.llamaindex import LlamaIndexKnowledgeBase

# Set page configuration
st.set_page_config(
    page_title="Medical-Legal Case Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up API keys using Streamlit secrets
try:
    # Try to get API keys from Streamlit secrets
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    # Fallback for local development (not recommended for production)
    GEMINI_API_KEY = ""  # Leave empty - will be set via UI
    openai_api_key = ""  # Leave empty - will be set via UI

# Set environment variables if keys are available
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize session state variables
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'case_factors' not in st.session_state:
    st.session_state.case_factors = {}
if 'probability' not in st.session_state:
    st.session_state.probability = 0
if 'adjustments' not in st.session_state:
    st.session_state.adjustments = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'critical_events' not in st.session_state:
    st.session_state.critical_events = {}
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'initial_query' not in st.session_state:
    st.session_state.initial_query = "Provide a comprehensive analysis of this case focusing on critical events and quality of care issues."
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
    
# Function to handle form submission
def set_form_submitted():
    st.session_state.form_submitted = True


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


# Trial Loss Probability Calculator functions
def calculate_trial_loss_probability(case_factors):
    """Calculate probability of losing a medical malpractice trial based on key case factors."""
    base_probability = 50
    adjustments = []
    
    # Positive factors (decrease probability)
    if case_factors.get("patient_left_ama", False):
        adjustment = -25
        base_probability += adjustment
        adjustments.append(("Patient left AMA", adjustment))
    
    if case_factors.get("refused_palliative_care", False):
        adjustment = -15
        base_probability += adjustment
        adjustments.append(("Refused palliative care", adjustment))
    
    if case_factors.get("documented_poor_prognosis", False):
        adjustment = -10
        base_probability += adjustment
        adjustments.append(("Documented poor prognosis", adjustment))
    
    if case_factors.get("multidisciplinary_consultation", False):
        adjustment = -5
        base_probability += adjustment
        adjustments.append(("Multidisciplinary consultation", adjustment))
    
    if case_factors.get("attempted_aggressive_care", False):
        adjustment = -5
        base_probability += adjustment
        adjustments.append(("Attempted aggressive care", adjustment))
    
    # Negative factors (increase probability)
    if case_factors.get("incomplete_documentation", False):
        adjustment = +15
        base_probability += adjustment
        adjustments.append(("Incomplete documentation", adjustment))
    
    if case_factors.get("delayed_care", False):
        adjustment = +20
        base_probability += adjustment
        adjustments.append(("Delayed care", adjustment))
    
    if case_factors.get("missed_diagnosis", False):
        adjustment = +25
        base_probability += adjustment
        adjustments.append(("Missed diagnosis", adjustment))
    
    # Care deviation factors - only increases probability if specifically identified
    care_deviations = {
        "icu_care_deviation": (+15, "ICU care deviation"),
        "shock_management_deviation": (+20, "Shock management deviation"),
        "respiratory_care_deviation": (+20, "Respiratory care deviation"),
        "inadequate_monitoring": (+15, "Inadequate monitoring")
    }
    
    for factor, (value, name) in care_deviations.items():
        if case_factors.get(factor, False):
            base_probability += value
            adjustments.append((name, value))
    
    # Floor/cap the probability
    final_probability = max(min(base_probability, 95), 15)
    
    return final_probability, adjustments


def extract_case_factors(medical_record_text):
    """Extract both case factors for probability calculation and critical events for reporting."""
    case_factors = {
        # Standard factors affecting probability
        "patient_left_ama": False,
        "refused_palliative_care": False,
        "documented_poor_prognosis": False,
        "multidisciplinary_consultation": False,
        "attempted_aggressive_care": False,
        "incomplete_documentation": False,
        "delayed_care": False,
        "missed_diagnosis": False,
        # Care deviation factors (affect probability)
        "icu_care_deviation": False,
        "shock_management_deviation": False,
        "respiratory_care_deviation": False,
        "inadequate_monitoring": False,
        # Critical events (for detection only, don't affect probability)
        "icu_stay": False,
        "shock_state": False,
        "vasopressor_use": False,
        "hypoxia": False,
        "aki": False,
        "code_blue": False,
        "intubation": False,
        "sepsis": False
    }
    
    # Define keywords for each factor
    keywords = {
        # Standard factors
        "patient_left_ama": ["against medical advice", "left AMA", "sign out AMA", "discharge AMA"],
        "refused_palliative_care": ["refused palliative", "declined palliative", "palliative care offered but", 
                                   "hospice candidate but", "refused hospice", "declined hospice"],
        "documented_poor_prognosis": ["poor prognosis", "guarded prognosis", "terminal", "end-stage", 
                                     "limited life expectancy", "high mortality risk", "high risk of death"],
        "multidisciplinary_consultation": ["multidisciplinary", "multiple consults", "team approach", 
                                          "consulted with", "consult", "specialist", "nephrology"],
        "attempted_aggressive_care": ["aggressive measures", "attempted intervention", "multiple attempts", 
                                     "exhaustive care", "extensive treatment"],
        "incomplete_documentation": ["incomplete documentation", "poor documentation", "inadequate documentation", 
                                    "missing documentation", "documentation gaps"],
        "delayed_care": ["delayed care", "delay in treatment", "untimely intervention", 
                        "treatment delay", "postponed care"],
        "missed_diagnosis": ["missed diagnosis", "failure to diagnose", "misdiagnosis", 
                            "diagnostic error", "delayed diagnosis"],
        # Care deviation factors
        "icu_care_deviation": ["inadequate ICU monitoring", "ICU transfer delay", "staffing issues in ICU", 
                              "insufficient critical care", "inappropriate ICU discharge"],
        "shock_management_deviation": ["inadequate fluid resuscitation", "delayed vasopressor administration", 
                                      "inappropriate vasopressor selection", "failure to recognize shock", 
                                      "inadequate treatment of shock"],
        "respiratory_care_deviation": ["delayed intubation", "inappropriate ventilator settings", 
                                      "failure to recognize hypoxia", "inadequate oxygen therapy"],
        "inadequate_monitoring": ["inadequate monitoring", "infrequent vital signs", "missed deterioration", 
                                 "failure to reassess"],
        # Critical events
        "icu_stay": ["ICU", "intensive care", "critical care", "step-down unit", "CCU"],
        "shock_state": ["shock", "hypotension", "hypotensive", "hemodynamic instability", "cardiogenic shock"],
        "vasopressor_use": ["vasopressor", "norepinephrine", "norepi", "epinephrine", "dopamine", 
                           "vasopressin", "levophed", "pressors"],
        "hypoxia": ["hypoxia", "hypoxic", "oxygen sat", "O2 sat", "SpO2", "desaturation", 
                   "respiratory distress", "respiratory failure"],
        "aki": ["AKI", "acute kidney injury", "renal failure", "creatinine elevation", 
               "kidney injury", "elevated BUN", "elevated creatinine"],
        "code_blue": ["code blue", "cardiac arrest", "cardiopulmonary arrest", "CPR", 
                     "resuscitation", "ACLS", "defibrillation", "ventricular fibrillation"],
        "intubation": ["intubation", "intubated", "mechanical ventilation", "ventilator", 
                      "endotracheal tube", "ETT", "respiratory failure"],
        "sepsis": ["sepsis", "septic", "bacteremia", "systemic inflammatory response", 
                  "SIRS", "septic shock", "source of infection"]
    }
    
    # Check for each keyword in the text
    text_lower = medical_record_text.lower()
    for factor, word_list in keywords.items():
        for keyword in word_list:
            if keyword.lower() in text_lower:
                case_factors[factor] = True
                break
    
    return case_factors


def setup_knowledge_base(pdf_content):
    """Set up knowledge base from PDF document."""
    try:
        # Create a unique ID for this knowledge base
        kb_id = f"kb_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create a temporary file for the PDF content
        temp_file = os.path.join(st.session_state.temp_dir, f"temp_{kb_id}.pdf")
        with open(temp_file, "wb") as f:
            f.write(pdf_content)
        
        # Process the document
        documents = SimpleDirectoryReader(input_files=[temp_file]).load_data()
        
        # Split text into chunks
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        
        # Create vector store index
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        
        # Create retriever and knowledge base
        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        knowledge_base = LlamaIndexKnowledgeBase(retriever=retriever)
        
        return knowledge_base, index, True
        
    except Exception as e:
        st.error(f"Error setting up knowledge base: {str(e)}")
        return None, None, False


# Function to display critical events
def display_critical_events(critical_events):
    """Display detected critical events with visual indicators."""
    st.subheader("Critical Medical Events Detected")
    
    # Create columns for cleaner display
    col1, col2 = st.columns(2)
    
    events = [
        "icu_stay", "shock_state", "vasopressor_use", "hypoxia", 
        "aki", "code_blue", "intubation", "sepsis"
    ]
    
    event_names = {
        "icu_stay": "ICU Stay",
        "shock_state": "Shock State",
        "vasopressor_use": "Vasopressor Use",
        "hypoxia": "Hypoxia",
        "aki": "Acute Kidney Injury",
        "code_blue": "Code Blue/Cardiac Arrest",
        "intubation": "Intubation",
        "sepsis": "Sepsis"
    }
    
    # Display events in two columns
    for i, event in enumerate(events):
        event_name = event_names.get(event, event.replace("_", " ").title())
        detected = critical_events.get(event, False)
        
        if i < 4:  # First 4 events in column 1
            with col1:
                if detected:
                    st.success(f"✓ {event_name}")
                else:
                    st.error(f"✗ {event_name}")
        else:  # Remaining events in column 2
            with col2:
                if detected:
                    st.success(f"✓ {event_name}")
                else:
                    st.error(f"✗ {event_name}")


# Function to display probability results
def display_probability_results(probability, adjustments):
    """Display trial loss probability with explanation of contributing factors."""
    st.subheader("Trial Loss Probability Assessment")
    
    # Create a progress bar for visual representation
    progress_color = "green" if probability < 40 else "orange" if probability < 70 else "red"
    st.progress(probability/100)
    
    # Display probability as a large number
    st.markdown(f"<h2 style='text-align: center; color: {progress_color};'>{probability}%</h2>", unsafe_allow_html=True)
    
    # Risk category
    risk_category = "Low Risk" if probability <= 30 else "Moderate Risk" if probability <= 60 else "High Risk" if probability <= 85 else "Very High Risk"
    st.markdown(f"<h4 style='text-align: center;'>{risk_category}</h4>", unsafe_allow_html=True)
    
    # Show factors affecting probability
    if adjustments:
        st.subheader("Contributing Factors")
        
        col1, col2 = st.columns(2)
        
        # Separate positive and negative factors
        positive_factors = [(factor, adj) for factor, adj in adjustments if adj < 0]
        negative_factors = [(factor, adj) for factor, adj in adjustments if adj > 0]
        
        with col1:
            st.markdown("#### Factors Decreasing Probability")
            if positive_factors:
                for factor, adjustment in positive_factors:
                    st.markdown(f"- {factor}: **{abs(adjustment)}%** decrease")
            else:
                st.markdown("*No factors decreasing probability*")
                
        with col2:
            st.markdown("#### Factors Increasing Probability")
            if negative_factors:
                for factor, adjustment in negative_factors:
                    st.markdown(f"- {factor}: **{adjustment}%** increase")
            else:
                st.markdown("*No factors increasing probability*")
    
    # Display explanation based on probability range
    st.subheader("Probability Interpretation")
    if probability <= 25:
        st.success("This represents a relatively low risk case with strong defense factors.")
    elif probability <= 50:
        st.info("This represents a moderate risk case with some positive defense factors.")
    elif probability <= 75:
        st.warning("This represents a significant risk case with concerning factors.")
    else:
        st.error("This represents a high risk case with multiple negative factors.")


def get_analysis_prompt(query, analysis_type, probability=None, adjustments=None):
    """Generate AI analysis prompt based on selected type and case data."""
    
    # Create probability guidance
    probability_guidance = ""
    if probability is not None and adjustments is not None:
        probability_guidance = f"""
        IMPORTANT: Based on hierarchical factor analysis, the trial loss probability 
        is estimated at {probability}%. 
        
        The following factors were considered:
        """
        for factor_name, adjustment in adjustments:
            direction = "decreases" if adjustment < 0 else "increases"
            probability_guidance += f"- {factor_name}: {direction} probability by {abs(adjustment)}%\n"
        
        probability_guidance += """
        Note: The mere presence of critical medical events (ICU stays, vasopressor use, etc.) 
        does not automatically increase probability. Only deviations from the standard of care 
        in managing these events affect the trial loss probability.
        """
    
    # Build prompt based on analysis type
    if analysis_type == "Mortality/Quality Review":
        prompt = f"""
        # Mortality/Quality Review Analysis Request
        
        Question: {query}
        
        IMPORTANT: SEARCH THE KNOWLEDGE BASE FOR RELEVANT INFORMATION BEFORE ANSWERING.
        
        Format your analysis with these exact section headings:
        
        ## Case Summary
        - Provide a concise 1-2 paragraph overview of the case
        - Include key patient demographics, primary medical issues, and outcome
        - Summarize the most significant events and care concerns in the case
        
        ## Critical Events Identification
        - Identify all critical medical events: ICU stays, shock states, vasopressor use, hypoxic events, AKI, code events
        - Provide specific details about each event (timing, severity, response)
        
        ## Key Clinical Events Timeline
        - Create chronological timeline with critical decision points
        - Note specific vital signs, lab values, and clinical findings
        
        ## Standard of Care Assessment [Score: X/5]
        - Focus on management of critical clinical events rather than routine care
        - Evaluate timeliness and appropriateness of interventions during deterioration
        
        ## Preventability Analysis [Classification: X]
        - Classify as: Certainly preventable, Probably preventable, Probably not preventable, or Certainly not preventable
        - Identify specific intervention points that could have changed outcome
        
        ## Improvement Recommendations
        - Propose specific improvements to prevent similar outcomes
        """
    
    elif analysis_type == "Medical-Legal Review":
        prompt = f"""
        # Medical-Legal Review Analysis Request
        
        Question: {query}
        
        IMPORTANT: SEARCH THE KNOWLEDGE BASE FOR RELEVANT INFORMATION BEFORE ANSWERING.
        
        {probability_guidance}
        
        Format your analysis with these exact section headings:
        
        ## Case Summary
        - Provide a concise 1-2 paragraph overview of the case
        - Include key patient demographics, primary medical issues, and outcome
        - Summarize the most significant events and care concerns in the case
        
        ## Medical Analysis
        ### Critical Events Identification
        - Identify all critical medical events: ICU stays, shock states, vasopressor use, hypoxic events, AKI, code events
        - Provide specific details about each event (timing, severity, response)
        
        ### Standard of Care Assessment [Score: X/5]
        - Focus on management of critical clinical events
        - Evaluate timeliness and appropriateness of interventions
        
        ## Legal Analysis
        ### Medical-Legal Risk Assessment [Trial Loss Probability: {probability if probability is not None else 'X'}%]
        - Explain how each identified factor contributes to this assessment
        - Evaluate the causation link between deviations and patient outcome
        - Note: Critical events alone don't indicate malpractice
        
        ### Legal Risk Scoring Matrix
           - Causation evidence strength: [score 1-5] - [Reasoning]
           - Documentation quality: [score 1-5] - [Reasoning]
           - Deviation severity: [score 1-5] - [Reasoning]
           - Jury sympathy factor: [score 1-5] - [Reasoning]
        
        ### Expert Witness Considerations
        - Identify at least 3 specific specialties needed for defense/plaintiff
        """
    
    else:  # Comprehensive
        prompt = f"""
        # Comprehensive Medical-Legal Analysis Request
        
        Question: {query}
        
        IMPORTANT: SEARCH THE KNOWLEDGE BASE FOR RELEVANT INFORMATION BEFORE ANSWERING.
        
        {probability_guidance}
        
        Format your analysis with these exact section headings:
        
        ## Case Summary
        - Provide a concise 1-2 paragraph overview of the case
        - Include key patient demographics, primary medical issues, and outcome
        - Summarize the most significant events and care concerns in the case
        
        ## Medical Analysis
        ### Critical Events Identification
        - Identify all critical medical events: ICU stays, shock states, vasopressor use, hypoxic events, AKI, code events
        - Provide specific details about each event (timing, severity, response)
        
        ### Key Clinical Events Timeline
        - Create chronological timeline with critical decision points
        - Note specific vital signs, lab values, and clinical findings
        
        ### Standard of Care Assessment [Score: X/5]
        - Focus on management of critical clinical events rather than routine care
        - Evaluate timeliness and appropriateness of interventions during deterioration
        
        ### Preventability Analysis [Classification: X]
        - Classify as: Certainly preventable, Probably preventable, Probably not preventable, or Certainly not preventable
        - Identify specific intervention points that could have changed outcome
        
        ## Legal Analysis
        ### Medical-Legal Risk Assessment [Trial Loss Probability: {probability if probability is not None else 'X'}%]
        - Explain how each identified factor contributes to this assessment
        - Evaluate the causation link between deviations and patient outcome
        - Note: Critical events alone don't indicate malpractice 
        
        ### Legal Risk Scoring Matrix
           - Causation evidence strength: [score 1-5] - [Reasoning]
           - Documentation quality: [score 1-5] - [Reasoning]
           - Deviation severity: [score 1-5] - [Reasoning]
           - Jury sympathy factor: [score 1-5] - [Reasoning]
        
        ### Expert Witness Considerations
        - Identify at least 3 specific specialties needed for defense/plaintiff
        """
    
    # Run the analysis
    final_prompt = f"""IMPORTANT: SEARCH THE KNOWLEDGE BASE FOR RELEVANT INFORMATION before responding.
    
    CRITICAL: For larger documents (>10MB), make sure to extract and retain key information about the patient, their condition, treatments, and outcomes in your response. Pay special attention to including demographic information, primary diagnoses, procedures performed, and final outcomes.
    
    {prompt}
    
    Remember to search the knowledge base for relevant information about this case.
    """
    
    return final_prompt


def run_chat_query(query, is_initial_analysis=False):
    """Process a chat query using the agent."""
    if st.session_state.agent is None:
        st.error("Agent not initialized. Please upload a document first.")
        return None
    
    try:
        # If this is the initial analysis, use a structured prompt
        if is_initial_analysis:
            prompt = get_analysis_prompt(
                query, 
                st.session_state.analysis_type,
                st.session_state.probability, 
                st.session_state.adjustments
            )
        else:
            # For follow-up questions, use a more conversational prompt with more context
            prompt = f"""
            Question: {query}
            
            IMPORTANT: 
            1. SEARCH THE KNOWLEDGE BASE for relevant information related to this case.
            2. Maintain context from our previous conversation about this medical case.
            3. Provide a focused, conversational response that addresses the specific question.
            4. If the question relates to medical factors, case analysis, or legal considerations,
               provide specific details from the document.
            5. For questions about case details, ALWAYS search the knowledge base first and include
               specific information from the document in your response.
            6. When asked for case summaries or details, include relevant patient information, medical
               conditions, treatments, outcomes, and any notable events from the document.
            
            Previous context from our conversation:
            {" ".join([message['content'][:300] + "..." for message in st.session_state.chat_history[-4:] if message['role'] == 'assistant'])}
            """
            
        # Get response from agent
        response = st.session_state.agent.run(prompt)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})
        
        return response.content
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return f"An error occurred: {str(e)}"


def create_agent(knowledge_base):
    """Create the Agno agent with the given knowledge base."""
    try:
        agent = Agent(
            name="Medical-Legal Expert",
            model=Gemini(id="gemini-2.0-flash-exp", api_key=GEMINI_API_KEY),
            knowledge=knowledge_base,
            search_knowledge=True,
            description="Expert medical and legal analyst specializing in medical malpractice case review, with focus on critical clinical events and root causes of patient deterioration.",
            instructions=[
                # Critical event focus instructions
                "PRIORITIZE identifying and analyzing root cause events in the clinical course.",
                "Pay special attention to critical clinical events: ICU stays, shock states, vasopressor use, hypoxia, AKI, codes/arrests.",
                "Create a detailed timeline of key physiologic deterioration with specific attention to vital sign changes.",
                "When examining standard of care, focus on management of critical events rather than routine care.",
                "Evaluate the causation link between key clinical deviations and specific physiologic harm.",
                
                # Medical review instructions
                "Evaluate clinical decision-making, including critical decision points and their implications.",
                "Analyze documentation completeness, accuracy, and communication between providers.",
                "IMPORTANT: Assign a Standard of Care score (1-5 scale) with specific rationale.",
                "IMPORTANT: Classify preventability as: Certainly preventable, Probably preventable, Probably not preventable, or Certainly not preventable.",
                
                # Legal review instructions
                "Evaluate legal liability based on medical evidence and documentation.",
                "Assess documentation quality and missing/inadequate documentation.",
                "Use hierarchical factor analysis to determine trial loss probability.",
                "IMPORTANT: Score these factors on a 1-5 scale: Causation evidence strength, Documentation quality, Deviation severity, Jury sympathy factor.",
                "IMPORTANT: Identify at least 3 specific specialties needed for defense/plaintiff expertise."
            ],
            markdown=True
        )
        return agent
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return None


# Function to process the document
def process_document(pdf_content, analysis_type, initial_query):
    """Process the document and set up the session state."""
    try:
        # Extract text
        pdf_text = extract_text_from_pdf(io.BytesIO(pdf_content))
        
        if not pdf_text:
            st.error("Could not extract text from PDF. Please ensure it's a valid document.")
            return False
        
        # Setup knowledge base
        knowledge_base, index, success = setup_knowledge_base(pdf_content)
        
        if not success:
            st.error("Failed to set up knowledge base.")
            return False
            
        st.session_state.knowledge_base = knowledge_base
        
        # Auto-extract case factors
        st.session_state.case_factors = extract_case_factors(pdf_text)
        
        # Extract critical events (subset of case_factors)
        st.session_state.critical_events = {
            key: st.session_state.case_factors.get(key, False) 
            for key in ["icu_stay", "shock_state", "vasopressor_use", "hypoxia", 
                      "aki", "code_blue", "intubation", "sepsis"]
        }
        
        # Calculate probability
        st.session_state.probability, st.session_state.adjustments = calculate_trial_loss_probability(
            st.session_state.case_factors
        )
        
        # Create agent
        st.session_state.agent = create_agent(knowledge_base)
        
        # Set document processed flag
        st.session_state.document_processed = True
        st.session_state.analysis_type = analysis_type
        st.session_state.initial_query = initial_query
        
        # Start with a clean chat history
        st.session_state.chat_history = []
        
        return True
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False


# Main application interface
st.title("Medical-Legal Case Analysis")
st.markdown("*Interactive AI-powered analysis of medical malpractice cases*")

# Sidebar with settings and information
with st.sidebar:
    st.header("Settings")
    
    # API Key configuration (optional since they're already set)
    with st.expander("API Keys"):
        st.text_input("Google Gemini API Key", value=GEMINI_API_KEY, type="password", key="gemini_key")
        if st.button("Update API Key"):
            st.session_state.GEMINI_API_KEY = st.session_state.gemini_key
            os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_key
            st.success("API key updated")
    
    # Info about the app
    with st.expander("About this app"):
        st.markdown("""
        This app provides AI-powered analysis of medical malpractice cases.
        
        **Features:**
        - Document analysis with LlamaIndex and Agno
        - Critical event identification
        - Trial loss probability calculation
        - Medical and legal analysis
        - Interactive chat with document context
        
        Powered by Gemini 2.0 Flash model and agentic frameworks.
        """)
    
    # Case factor adjustment section
    if st.session_state.document_processed:
        with st.expander("Adjust Case Factors"):
            st.subheader("Standard Case Factors")
            standard_factors = {
                "patient_left_ama": "Patient Left AMA",
                "refused_palliative_care": "Refused Palliative Care",
                "documented_poor_prognosis": "Documented Poor Prognosis",
                "multidisciplinary_consultation": "Multidisciplinary Consultation",
                "attempted_aggressive_care": "Attempted Aggressive Care",
                "incomplete_documentation": "Incomplete Documentation",
                "delayed_care": "Delayed Care",
                "missed_diagnosis": "Missed Diagnosis"
            }
            
            care_deviation_factors = {
                "icu_care_deviation": "ICU Care Deviation",
                "shock_management_deviation": "Shock Management Deviation", 
                "respiratory_care_deviation": "Respiratory Care Deviation",
                "inadequate_monitoring": "Inadequate Monitoring"
            }
            
            # Display checkboxes for each factor
            for key, label in standard_factors.items():
                st.session_state.case_factors[key] = st.checkbox(
                    label, 
                    value=st.session_state.case_factors.get(key, False),
                    key=f"sidebar_{key}"
                )
            
            st.subheader("Care Deviation Factors")
            for key, label in care_deviation_factors.items():
                st.session_state.case_factors[key] = st.checkbox(
                    label, 
                    value=st.session_state.case_factors.get(key, False),
                    key=f"sidebar_{key}"
                )
            
            # Button to recalculate
            if st.button("Recalculate Probability", key="recalc_button"):
                st.session_state.probability, st.session_state.adjustments = calculate_trial_loss_probability(
                    st.session_state.case_factors
                )
                st.success(f"Probability recalculated: {st.session_state.probability}%")
                st.rerun()

# Main content area - Document upload or analysis display
if not st.session_state.document_processed:
    # Document upload section
    st.header("Document Upload")
    
    # Use a form to prevent automatic rerunning
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Upload medical record PDF", type="pdf", help="Upload a PDF containing the medical case to analyze")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Mortality/Quality Review", "Medical-Legal Review", "Comprehensive (Both)"],
                index=2,
                help="Choose the type of analysis to perform"
            )
        
        with col2:
            initial_query = st.text_input(
                "Initial analysis prompt", 
                value="Provide a comprehensive analysis of this case focusing on critical events and quality of care issues.",
                help="Enter your initial question to analyze the document"
            )
        
        # Process document button
        submitted = st.form_submit_button("Process Document", on_click=set_form_submitted)
    
    # Process the document if the form was submitted
    if st.session_state.form_submitted and uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save PDF content
            pdf_content = uploaded_file.read()
            
            # Process the document
            if process_document(pdf_content, analysis_type, initial_query):
                st.success("Document processed successfully!")
                st.session_state.form_submitted = False  # Reset form state
                st.rerun()
            else:
                st.error("Failed to process document. Please try again.")
                st.session_state.form_submitted = False  # Reset form state

else:
    # Analysis and chat interface after document is processed
    st.header("Case Analysis")
    
    # Display tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Case Summary", "Critical Events", "Probability Assessment", "Chat Interface"])
    
    # Extract case summary from chat history if available
    case_summary = ""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "assistant":
                content = message["content"]
                if "## Case Summary" in content:
                    summary_section = content.split("## Case Summary")[1].split("##")[0].strip()
                    case_summary = summary_section
                    break
    
    # Tab 1: Case Summary
    with tab1:
        st.subheader("Case Overview")
        
        if case_summary:
            st.markdown(case_summary)
        else:
            # If no summary is found in chat history, generate one
            if not st.session_state.chat_history:
                st.info("Case summary will be generated when initial analysis is complete.")
            else:
                st.info("No detailed case summary found. Please check the Chat Interface tab for the full analysis.")
                
        # Display a quick stats section
        st.subheader("Quick Case Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            critical_count = sum(1 for value in st.session_state.critical_events.values() if value)
            st.metric("Critical Events Detected", critical_count, f"{critical_count}/8")
            
        with col2:
            st.metric("Trial Loss Probability", f"{st.session_state.probability}%", 
                     delta="Low Risk" if st.session_state.probability <= 30 else 
                     "Moderate Risk" if st.session_state.probability <= 60 else 
                     "High Risk" if st.session_state.probability <= 85 else "Very High Risk")
            
        with col3:
            positive_factors = sum(1 for _, adj in st.session_state.adjustments if adj < 0)
            negative_factors = sum(1 for _, adj in st.session_state.adjustments if adj > 0)
            st.metric("Risk Factors", f"{negative_factors} negative, {positive_factors} positive")
    
    # Tab 2: Critical Events
    with tab2:
        display_critical_events(st.session_state.critical_events)
        
        standard_factors = {key: value for key, value in st.session_state.case_factors.items() 
                          if key not in st.session_state.critical_events and value == True}
        
        if standard_factors:
            st.subheader("Other Case Factors Detected")
            cols = st.columns(2)
            i = 0
            for key, value in standard_factors.items():
                col = cols[i % 2]
                with col:
                    factor_name = key.replace('_', ' ').title()
                    st.success(f"✓ {factor_name}")
                i += 1
    
    # Tab 3: Probability Assessment
    with tab3:
        display_probability_results(st.session_state.probability, st.session_state.adjustments)
    
    # Tab 4: Chat Interface
    with tab4:
        st.subheader("Chat with Document")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
        
        # Initial analysis (only if chat history is empty)
        if not st.session_state.chat_history:
            with st.spinner("Generating initial analysis..."):
                initial_analysis = run_chat_query(st.session_state.initial_query, is_initial_analysis=True)
                if initial_analysis:
                    st.markdown(f"**AI:** {initial_analysis}")
                    # Rerun to update the case summary tab
                    st.rerun()
        
        # Chat input
        with st.form("chat_form"):
            user_query = st.text_input("Ask a question about this case:", key="user_query")
            send_pressed = st.form_submit_button("Send")
            
        if send_pressed and user_query:
            with st.spinner("Analyzing..."):
                # Get response
                response = run_chat_query(user_query)
                
                # Display response
                st.markdown(f"**You:** {user_query}")
                st.markdown(f"**AI:** {response}")
                
                # Scroll to bottom (using JavaScript)
                js = f"""
                <script>
                    function scroll_to_bottom() {{
                        var chatContainer = window.parent.document.querySelector('[data-testid="stVerticalBlock"]');
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }}
                    scroll_to_bottom();
                </script>
                """
                st.components.v1.html(js, height=0)
