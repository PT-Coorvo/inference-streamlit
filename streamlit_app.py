# =============================================================================
# CV RECOMMENDATION SYSTEM - STREAMLIT APPLICATION
# Complete Web Interface untuk CV Screening & Recommendation
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, date
import re
from typing import List, Dict
import io

# Import system classes (pastikan file system ada di folder yang sama)
try:
    from cv_system import CompleteCVRecommendationSystem, get_hybrid_recommendations, load_model, main as train_system
    SYSTEM_AVAILABLE = True
except ImportError:
    SYSTEM_AVAILABLE = False

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CV Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .candidate-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    
    .score-excellent { color: #28a745; font-weight: bold; }
    .score-good { color: #17a2b8; font-weight: bold; }
    .score-average { color: #ffc107; font-weight: bold; }
    .score-poor { color: #dc3545; font-weight: bold; }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_system_model(uploaded_file=None, file_path=None):
    """Load CV system model"""
    try:
        if uploaded_file is not None:
            # Load from uploaded file
            model_data = pickle.load(uploaded_file)
            cv_system = CompleteCVRecommendationSystem()
            
            # Restore components
            cv_system.model_version = model_data['model_version']
            cv_system.training_date = model_data['training_date']
            cv_system.candidates_data = model_data['candidates_data']
            cv_system.skills_tfidf_vectorizer = model_data['skills_tfidf_vectorizer']
            cv_system.skills_tfidf_matrix = model_data['skills_tfidf_matrix']
            cv_system.content_tfidf_vectorizer = model_data['content_tfidf_vectorizer']
            cv_system.content_tfidf_matrix = model_data['content_tfidf_matrix']
            cv_system.user_item_matrix = model_data['user_item_matrix']
            cv_system.svd_model = model_data['svd_model']
            cv_system.user_factors = model_data['user_factors']
            cv_system.item_factors = model_data['item_factors']
            cv_system.predicted_interactions = model_data['predicted_interactions']
            cv_system.job_profiles = model_data['job_profiles']
            
            return cv_system
            
        elif file_path:
            # Load from file path
            return load_model(file_path)
        else:
            return None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_score_color_class(score):
    """Get CSS class based on score"""
    if score >= 80:
        return "score-excellent"
    elif score >= 60:
        return "score-good"
    elif score >= 40:
        return "score-average"
    else:
        return "score-poor"

def generate_chart_key(chart_type, *identifiers):
    """Generate unique key for charts to avoid duplicate element ID errors"""
    import hashlib
    key_string = f"{chart_type}_{'_'.join(str(id).replace(' ', '_').replace(',', '_') for id in identifiers)}"
    # Truncate if too long and add hash for uniqueness
    if len(key_string) > 50:
        hash_suffix = hashlib.md5(key_string.encode()).hexdigest()[:8]
        key_string = key_string[:42] + hash_suffix
    return key_string

def format_salary(salary):
    """Format salary untuk display"""
    if salary >= 1000000:
        return f"Rp {salary/1000000:.1f}M"
    else:
        return f"Rp {salary:,.0f}"

def calculate_age_from_date(birth_date):
    """Calculate age from birth date"""
    if birth_date:
        today = date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return max(16, min(65, age))
    return 25

def extract_experience_years(experience_text):
    """Extract experience years from text"""
    if pd.isna(experience_text) or not experience_text:
        return 0
    years = re.findall(r'(\d+)\s*(?:tahun|years?)', str(experience_text).lower())
    return int(years[0]) if years else 0

def process_skills_input(skills_input):
    """Process skills input and return cleaned list"""
    if not skills_input:
        return []
    
    # Split by comma and clean each skill
    skills_list = [skill.strip() for skill in skills_input.split(',') if skill.strip()]
    return skills_list

def validate_email(email):
    """Validate email format"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

def add_candidate_to_system(cv_system, candidate_data):
    """
    Add new candidate to the system and update models
    
    Args:
        cv_system: CV system instance
        candidate_data: Dictionary containing candidate information
        
    Returns:
        bool: Success status
    """
    try:
        # Check if email already exists
        existing_emails = cv_system.candidates_data['Email address'].str.lower()
        if candidate_data['email'].lower() in existing_emails.values:
            return False, "Email already exists in database"
        
        # Process candidate data
        processed_candidate = create_candidate_record(candidate_data)
        
        # Add to dataframe
        new_row = pd.DataFrame([processed_candidate])
        cv_system.candidates_data = pd.concat([cv_system.candidates_data, new_row], ignore_index=True)
        
        # Update models
        update_system_models(cv_system)
        
        return True, "Candidate added successfully"
        
    except Exception as e:
        return False, f"Error adding candidate: {str(e)}"

def create_candidate_record(candidate_data):
    """Create a properly formatted candidate record"""
    
    # Calculate age
    age = calculate_age_from_date(candidate_data.get('birth_date'))
    
    # Extract experience years
    experience_years = extract_experience_years(candidate_data.get('experience', ''))
    
    # Process skills
    skills_list = process_skills_input(candidate_data.get('skills_input', ''))
    skills_text = ', '.join(skills_list)
    skills_cleaned = ' '.join([skill.lower().strip() for skill in skills_list])
    
    # Create combined text for TF-IDF
    combined_text = f"{skills_cleaned} {candidate_data.get('desired_position', '').lower()} {candidate_data.get('faculty_major', '').lower()}"
    
    # Format birth date
    birth_date_str = ""
    if candidate_data.get('birth_date'):
        birth_date_str = candidate_data['birth_date'].strftime('%Y-%m-%d')
    
    # Create the candidate record
    candidate_record = {
        'Full name': candidate_data.get('full_name', ''),
        'Email address': candidate_data.get('email', ''),
        'WhatsApp number': candidate_data.get('phone', ''),
        'Date of birth': birth_date_str,
        'Desired positions': candidate_data.get('desired_position', ''),
        'Experience': candidate_data.get('experience', 'Fresh Graduate'),
        'Highest formal of education': candidate_data.get('education_level', 'Bachelor'),
        'Faculty/Major': candidate_data.get('faculty_major', ''),
        'Skills': skills_text,
        'Expected salary (IDR)': candidate_data.get('expected_salary', 8000000),
        'Current address': candidate_data.get('current_address', ''),
        'Current status': candidate_data.get('current_status', 'Available'),
        
        # Processed fields
        'age': age,
        'experience_years': experience_years,
        'expected_salary_normalized': candidate_data.get('expected_salary', 8000000),
        'skills_list': skills_list,
        'skills_count': len(skills_list),
        'skills_text': ' '.join(skills_list),
        'skills_cleaned': skills_cleaned,
        'combined_text': combined_text
    }
    
    return candidate_record

def update_system_models(cv_system):
    """Update TF-IDF and collaborative filtering models after adding new candidate"""
    try:
        # Update skills TF-IDF
        skills_texts = cv_system.candidates_data['skills_cleaned'].fillna('')
        cv_system.skills_tfidf_matrix = cv_system.skills_tfidf_vectorizer.fit_transform(skills_texts)
        
        # Update content TF-IDF
        combined_texts = cv_system.candidates_data['combined_text'].fillna('')
        cv_system.content_tfidf_matrix = cv_system.content_tfidf_vectorizer.fit_transform(combined_texts)
        
        # Update collaborative filtering if enough data
        if len(cv_system.candidates_data) > 10:
            update_collaborative_filtering_model(cv_system)
        
    except Exception as e:
        st.warning(f"Warning: Could not update all models: {str(e)}")

def update_collaborative_filtering_model(cv_system):
    """Update collaborative filtering model"""
    try:
        from sklearn.decomposition import TruncatedSVD
        
        # Recreate user-item matrix
        candidates_skills = cv_system.candidates_data['skills_list'].tolist()
        n_candidates = len(candidates_skills)
        
        # Get all unique skills
        all_skills = set()
        for skills_list in candidates_skills:
            all_skills.update([skill.lower().strip() for skill in skills_list])
        
        all_skills = list(all_skills)
        
        # Create binary interaction matrix
        interaction_matrix = np.zeros((n_candidates, len(all_skills)))
        
        for i, skills_list in enumerate(candidates_skills):
            for skill in skills_list:
                skill_clean = skill.lower().strip()
                if skill_clean in all_skills:
                    j = all_skills.index(skill_clean)
                    interaction_matrix[i][j] = 1
        
        cv_system.user_item_matrix = pd.DataFrame(
            interaction_matrix,
            index=range(n_candidates),
            columns=all_skills
        )
        
        # Update SVD model
        n_components = min(50, min(cv_system.user_item_matrix.shape) - 1)
        
        if n_components >= 1:
            cv_system.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            cv_system.user_factors = cv_system.svd_model.fit_transform(cv_system.user_item_matrix)
            cv_system.item_factors = cv_system.svd_model.components_
            cv_system.predicted_interactions = np.dot(cv_system.user_factors, cv_system.item_factors)
            
    except Exception as e:
        st.warning(f"Warning: Could not update collaborative filtering: {str(e)}")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 CV Recommendation System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'cv_system' not in st.session_state:
        st.session_state.cv_system = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = {}
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Control")
        
        # Model Management
        st.subheader("📁 Model Management")
        
        # Option 1: Upload model file
        uploaded_file = st.file_uploader(
            "Upload Model File (.pkl)", 
            type=['pkl'],
            help="Upload pre-trained CV recommendation model"
        )
        
        if uploaded_file and st.button("Load Uploaded Model"):
            with st.spinner("Loading model..."):
                cv_system = load_system_model(uploaded_file=uploaded_file)
                if cv_system:
                    st.session_state.cv_system = cv_system
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
        
        # Option 2: Load default model
        if st.button("Load Default Model"):
            with st.spinner("Loading default model..."):
                cv_system = load_system_model(file_path='cv_recommendation_model.pkl')
                if cv_system:
                    st.session_state.cv_system = cv_system
                    st.success("✅ Default model loaded!")
                    st.rerun()
        
        # Option 3: Train new model
        st.subheader("🤖 Train New Model")
        if st.button("Train New Model"):
            if SYSTEM_AVAILABLE:
                with st.spinner("Training new model... This may take a few minutes."):
                    try:
                        cv_system, job_profiles = train_system(save_model_flag=True)
                        st.session_state.cv_system = cv_system
                        st.success("✅ New model trained successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
            else:
                st.error("System modules not available for training.")
        
        # Model Status
        if st.session_state.cv_system:
            st.success("🟢 Model Ready")
            cv_system = st.session_state.cv_system
            
            st.write("**Model Info:**")
            st.write(f"- Version: {cv_system.model_version}")
            st.write(f"- Training Date: {cv_system.training_date.strftime('%Y-%m-%d %H:%M') if cv_system.training_date else 'Unknown'}")
            st.write(f"- Candidates: {len(cv_system.candidates_data)}")
            st.write(f"- Job Profiles: {len(cv_system.job_profiles)}")
        else:
            st.warning("🔴 No Model Loaded")
    
    # Main Content
    if st.session_state.cv_system is None:
        # Welcome Page
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            ## 👋 Welcome to CV Recommendation System
            
            ### 🚀 Features:
            - **Hybrid AI Recommendation** (Content-Based + Collaborative + Skills Matching)
            - **Interactive Job Profiles** with customizable requirements
            - **Real-time Analytics** dan performance insights
            - **Manual Candidate Entry** untuk data input langsung
            - **Export Results** dalam multiple formats
            - **Model Persistence** untuk reusability
            
            ### 📋 Getting Started:
            1. **Load a Model** dari sidebar (upload .pkl atau train new)
            2. **Add Candidates** secara manual atau bulk import
            3. **Select Job Profile** yang ingin direkrut
            4. **Generate Recommendations** dan analyze results
            5. **Export** results untuk HR team
            
            ### 🎯 System Architecture:
            - **Skills Matching**: 40% weight
            - **Content-Based**: 25% weight  
            - **Collaborative Filtering**: 20% weight
            - **Feature-Based**: 15% weight
            """)
        
        return
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Recommendations", 
        "📊 Analytics", 
        "👥 Candidates",
        "➕ Add Candidate",
        "💼 Job Profiles",
        "⚙️ Settings"
    ])
    
    cv_system = st.session_state.cv_system
    
    # =============================================================================
    # TAB 1: RECOMMENDATIONS
    # =============================================================================
    with tab1:
        st.header("🎯 Get Job Recommendations")
        
        # Job Selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            job_options = {
                job_id: profile['title'] 
                for job_id, profile in cv_system.job_profiles.items()
            }
            
            selected_job_id = st.selectbox(
                "Select Job Position:",
                options=list(job_options.keys()),
                format_func=lambda x: job_options[x],
                help="Choose the job position untuk generate recommendations"
            )
        
        with col2:
            top_k = st.slider("Number of Recommendations:", 1, 10, 5)
        
        # Job Details
        if selected_job_id:
            job_profile = cv_system.job_profiles[selected_job_id]
            
            with st.expander(f"📋 {job_profile['title']} - Job Details", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Description:**")
                    st.write(job_profile['description'])
                    
                    st.write("**Required Skills:**")
                    for skill in job_profile['required_skills']:
                        st.write(f"• {skill}")
                
                with col2:
                    st.write("**Preferred Skills:**")
                    for skill in job_profile['preferred_skills']:
                        st.write(f"• {skill}")
                    
                    st.write("**Requirements:**")
                    st.write(f"• Experience: {job_profile['min_experience']}-{job_profile['max_experience']} years")
                    st.write(f"• Education: {job_profile['preferred_education']}")
                    
                    salary_min, salary_max = job_profile['salary_range']
                    st.write(f"• Salary: {format_salary(salary_min)} - {format_salary(salary_max)}")
        
        # Generate Recommendations Button
        if st.button("🚀 Generate Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                try:
                    recommendations = get_hybrid_recommendations(
                        cv_system, selected_job_id, top_k=top_k, debug=False
                    )
                    st.session_state.recommendations[selected_job_id] = recommendations
                    st.success(f"✅ Generated {len(recommendations)} recommendations!")
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    return
        
        # Display Recommendations
        if selected_job_id in st.session_state.recommendations:
            recommendations = st.session_state.recommendations[selected_job_id]
            
            st.header("📊 Recommendation Results")
            
            # Summary Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = np.mean([r['final_score'] for r in recommendations])
                st.markdown(f'<div class="metric-card"><h3>{avg_score:.1f}%</h3><p>Average Score</p></div>', unsafe_allow_html=True)
            
            with col2:
                top_score = max([r['final_score'] for r in recommendations])
                st.markdown(f'<div class="metric-card"><h3>{top_score:.1f}%</h3><p>Top Score</p></div>', unsafe_allow_html=True)
            
            with col3:
                avg_skills = np.mean([len(r['matched_required_skills']) for r in recommendations])
                st.markdown(f'<div class="metric-card"><h3>{avg_skills:.1f}</h3><p>Avg Matched Skills</p></div>', unsafe_allow_html=True)
            
            with col4:
                avg_exp = np.mean([r['experience_years'] for r in recommendations])
                st.markdown(f'<div class="metric-card"><h3>{avg_exp:.1f}</h3><p>Avg Experience</p></div>', unsafe_allow_html=True)
            
            # Detailed Results
            for i, rec in enumerate(recommendations):
                with st.expander(f"🏆 Rank {rec['rank']}: {rec['name']} - Score: {rec['final_score']}%", expanded=(i==0)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**📧 Email:** {rec['email']}")
                        st.write(f"**📞 Phone:** {rec['phone']}")
                        st.write(f"**🎯 Desired Position:** {rec['desired_position']}")
                        st.write(f"**🎓 Education:** {rec['education']} - {rec['major']}")
                        st.write(f"**💼 Experience:** {rec['experience_years']} years")
                        st.write(f"**💰 Expected Salary:** {format_salary(rec['expected_salary'])}")
                        st.write(f"**📍 Location:** {rec['address']}")
                        
                        # Skills Analysis
                        st.write("**✅ Matched Required Skills:**")
                        if rec['matched_required_skills']:
                            for skill in rec['matched_required_skills']:
                                st.write(f"• {skill}")
                        else:
                            st.write("None")
                        
                        st.write("**⭐ Matched Preferred Skills:**")
                        if rec['matched_preferred_skills']:
                            for skill in rec['matched_preferred_skills']:
                                st.write(f"• {skill}")
                        else:
                            st.write("None")
                    
                    with col2:
                        # Scoring Breakdown Chart
                        try:
                            scores = {
                                'Skills Match': rec['skills_match_score'],
                                'Content-Based': rec['content_similarity_score'],
                                'Collaborative': rec['collaborative_score'],
                                'Features': rec['feature_score']
                            }
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(scores.keys()),
                                    y=list(scores.values()),
                                    marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
                                )
                            ])
                            
                            fig.update_layout(
                                title=f"Score Breakdown",
                                yaxis_title="Score (%)",
                                height=300,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=generate_chart_key("score_breakdown", rec['rank'], selected_job_id, rec['name']))
                            
                            # Final Score Display
                            score_class = get_score_color_class(rec['final_score'])
                            st.markdown(f"<h2 class='{score_class}'>Final Score: {rec['final_score']}%</h2>", unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error displaying chart: {str(e)}")
                            st.write(f"**Final Score:** {rec['final_score']}%")
            
            # Export Options
            st.header("📥 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export to CSV
                df_export = pd.DataFrame([
                    {
                        'Rank': r['rank'],
                        'Name': r['name'],
                        'Email': r['email'],
                        'Phone': r['phone'],
                        'Education': r['education'],
                        'Experience': r['experience_years'],
                        'Expected_Salary': r['expected_salary'],
                        'Final_Score': r['final_score'],
                        'Skills_Match': r['skills_match_score'],
                        'Content_Score': r['content_similarity_score'],
                        'Collaborative_Score': r['collaborative_score'],
                        'Feature_Score': r['feature_score']
                    } for r in recommendations
                ])
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"recommendations_{selected_job_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export to JSON
                json_data = json.dumps(recommendations, indent=2, default=str)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"recommendations_{selected_job_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            
            with col3:
                # Generate Report
                if st.button("📄 Generate Report"):
                    report = generate_recommendation_report(job_profile, recommendations)
                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name=f"report_{selected_job_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
    
    # =============================================================================
    # TAB 2: ANALYTICS
    # =============================================================================
    with tab2:
        st.header("📊 System Analytics")
        
        if not st.session_state.recommendations:
            st.info("Generate some recommendations first to see analytics!")
            return
        
        # Overall Performance Metrics
        st.subheader("🎯 Overall Performance")
        
        all_recommendations = []
        for job_recs in st.session_state.recommendations.values():
            all_recommendations.extend(job_recs)
        
        if all_recommendations:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_candidates = len(set([r['name'] for r in all_recommendations]))
                st.metric("Total Candidates Evaluated", total_candidates)
            
            with col2:
                avg_final_score = np.mean([r['final_score'] for r in all_recommendations])
                st.metric("Average Final Score", f"{avg_final_score:.1f}%")
            
            with col3:
                avg_skills_match = np.mean([r['skills_match_score'] for r in all_recommendations])
                st.metric("Average Skills Match", f"{avg_skills_match:.1f}%")
            
            with col4:
                avg_experience = np.mean([r['experience_years'] for r in all_recommendations])
                st.metric("Average Experience", f"{avg_experience:.1f} years")
        
        # Score Distribution
        st.subheader("📈 Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Final Score Distribution
            final_scores = [r['final_score'] for r in all_recommendations]
            
            try:
                fig = px.histogram(
                    x=final_scores,
                    nbins=20,
                    title="Final Score Distribution",
                    labels={'x': 'Final Score (%)', 'y': 'Count'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="final_score_distribution")
            except Exception as e:
                st.error(f"Error displaying score distribution chart: {str(e)}")
                st.write(f"Final scores: {final_scores[:10]}...")  # Show sample data
        
        with col2:
            # Component Scores Comparison
            try:
                score_components = []
                for r in all_recommendations:
                    score_components.extend([
                        {'Component': 'Skills Match', 'Score': r['skills_match_score'], 'Candidate': r['name']},
                        {'Component': 'Content-Based', 'Score': r['content_similarity_score'], 'Candidate': r['name']},
                        {'Component': 'Collaborative', 'Score': r['collaborative_score'], 'Candidate': r['name']},
                        {'Component': 'Features', 'Score': r['feature_score'], 'Candidate': r['name']}
                    ])
                
                df_components = pd.DataFrame(score_components)
                
                fig = px.box(
                    df_components,
                    x='Component',
                    y='Score',
                    title="Score Components Distribution"
                )
                st.plotly_chart(fig, use_container_width=True, key="score_components_box")
            except Exception as e:
                st.error(f"Error displaying components chart: {str(e)}")
                st.write("Score components data temporarily unavailable")
        
        # Job-wise Performance
        st.subheader("💼 Job-wise Performance")
        
        job_performance = []
        for job_id, recs in st.session_state.recommendations.items():
            job_title = cv_system.job_profiles[job_id]['title']
            avg_score = np.mean([r['final_score'] for r in recs])
            top_score = max([r['final_score'] for r in recs])
            candidates_count = len(recs)
            
            job_performance.append({
                'Job': job_title,
                'Average Score': avg_score,
                'Top Score': top_score,
                'Candidates': candidates_count
            })
        
        df_job_perf = pd.DataFrame(job_performance)
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fig = px.bar(
                    df_job_perf,
                    x='Job',
                    y='Average Score',
                    title="Average Score by Job Position"
                )
                st.plotly_chart(fig, use_container_width=True, key="avg_score_by_job")
            except Exception as e:
                st.error(f"Error displaying average score chart: {str(e)}")
        
        with col2:
            try:
                fig = px.bar(
                    df_job_perf,
                    x='Job',
                    y='Top Score',
                    title="Top Score by Job Position"
                )
                st.plotly_chart(fig, use_container_width=True, key="top_score_by_job")
            except Exception as e:
                st.error(f"Error displaying top score chart: {str(e)}")
        
        # Skills Analysis
        st.subheader("🛠️ Skills Analysis")
        
        # Most Common Skills
        all_skills = []
        for r in all_recommendations:
            all_skills.extend(r['matched_required_skills'])
            all_skills.extend(r['matched_preferred_skills'])
        
        if all_skills:
            skills_count = pd.Series(all_skills).value_counts().head(10)
            
            try:
                fig = px.bar(
                    x=skills_count.values,
                    y=skills_count.index,
                    orientation='h',
                    title="Top 10 Most Matched Skills"
                )
                fig.update_layout(yaxis_title="Skills", xaxis_title="Match Count")
                st.plotly_chart(fig, use_container_width=True, key="top_matched_skills")
            except Exception as e:
                st.error(f"Error displaying skills chart: {str(e)}")
                st.write("**Top Skills:**")
                for skill, count in skills_count.head(5).items():
                    st.write(f"• {skill}: {count} matches")
    
    # =============================================================================
    # TAB 3: CANDIDATES
    # =============================================================================
    with tab3:
        st.header("👥 Candidate Database")
        
        # Candidate Statistics
        candidates_df = cv_system.candidates_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(candidates_df))
        with col2:
            avg_skills = candidates_df['skills_count'].mean()
            st.metric("Avg Skills per Candidate", f"{avg_skills:.1f}")
        with col3:
            avg_exp = candidates_df['experience_years'].mean()
            st.metric("Avg Experience", f"{avg_exp:.1f} years")
        with col4:
            avg_salary = candidates_df['expected_salary_normalized'].mean()
            st.metric("Avg Expected Salary", format_salary(avg_salary))
        
        # Filters
        st.subheader("🔍 Filter Candidates")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            exp_filter = st.slider(
                "Experience Years:",
                int(candidates_df['experience_years'].min()),
                int(candidates_df['experience_years'].max()),
                (int(candidates_df['experience_years'].min()), int(candidates_df['experience_years'].max()))
            )
        
        with col2:
            edu_options = ['All'] + list(candidates_df['Highest formal of education'].unique())
            edu_filter = st.selectbox("Education Level:", edu_options)
        
        with col3:
            skills_filter = st.number_input("Minimum Skills Count:", 0, int(candidates_df['skills_count'].max()), 0)
        
        # Apply filters
        filtered_df = candidates_df[
            (candidates_df['experience_years'] >= exp_filter[0]) &
            (candidates_df['experience_years'] <= exp_filter[1]) &
            (candidates_df['skills_count'] >= skills_filter)
        ]
        
        if edu_filter != 'All':
            filtered_df = filtered_df[filtered_df['Highest formal of education'] == edu_filter]
        
        st.write(f"**Showing {len(filtered_df)} of {len(candidates_df)} candidates**")
        
        # Candidate Table
        display_columns = [
            'Full name', 'Email address', 'Desired positions', 
            'Highest formal of education', 'experience_years', 
            'expected_salary_normalized', 'skills_count'
        ]
        
        display_df = filtered_df[display_columns].copy()
        display_df = display_df.rename(columns={
            'Full name': 'Name',
            'Email address': 'Email',
            'Desired positions': 'Desired Position',
            'Highest formal of education': 'Education',
            'experience_years': 'Experience (Years)',
            'expected_salary_normalized': 'Expected Salary',
            'skills_count': 'Skills Count'
        })
        
        # Format salary
        display_df['Expected Salary'] = display_df['Expected Salary'].apply(format_salary)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Candidate Details
        if st.button("🔍 View Detailed Candidate Analysis"):
            selected_candidate = st.selectbox(
                "Select Candidate:",
                options=filtered_df['Full name'].tolist()
            )
            
            if selected_candidate:
                candidate_data = filtered_df[filtered_df['Full name'] == selected_candidate].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Personal Information:**")
                    st.write(f"Name: {candidate_data['Full name']}")
                    st.write(f"Email: {candidate_data['Email address']}")
                    st.write(f"Phone: {candidate_data['WhatsApp number']}")
                    st.write(f"Address: {candidate_data['Current address']}")
                    st.write(f"Age: {candidate_data['age']} years")
                    
                    st.write("**Professional Information:**")
                    st.write(f"Desired Position: {candidate_data['Desired positions']}")
                    st.write(f"Experience: {candidate_data['experience_years']} years")
                    st.write(f"Education: {candidate_data['Highest formal of education']}")
                    st.write(f"Major: {candidate_data['Faculty/Major']}")
                    st.write(f"Expected Salary: {format_salary(candidate_data['expected_salary_normalized'])}")
                
                with col2:
                    st.write("**Skills:**")
                    skills_list = candidate_data['skills_list']
                    for skill in skills_list:
                        st.write(f"• {skill}")
                    
                    # Skills breakdown chart
                    if len(skills_list) > 0:
                        try:
                            skills_df = pd.DataFrame({'Skills': skills_list})
                            fig = px.bar(
                                y=skills_list[:10],  # Top 10 skills
                                orientation='h',
                                title=f"Skills Overview ({len(skills_list)} total)"
                            )
                            fig.update_layout(showlegend=False, height=400)
                            st.plotly_chart(fig, use_container_width=True, key=generate_chart_key("candidate_skills", selected_candidate))
                        except Exception as e:
                            st.error(f"Error displaying skills chart: {str(e)}")
                            st.write("**Skills List:**")
                            for skill in skills_list[:10]:
                                st.write(f"• {skill}")

    # =============================================================================
    # TAB 4: ADD CANDIDATE
    # =============================================================================
    with tab4:
        st.header("➕ Add New Candidate")
        st.write("Manually add a new candidate to the system database.")
        
        # Add Candidate Form
        with st.form("add_candidate_form", clear_on_submit=True):
            st.subheader("📝 Candidate Information")
            
            # Personal Information Section
            st.markdown("### 👤 Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                full_name = st.text_input(
                    "Full Name *", 
                    placeholder="Enter candidate's full name",
                    help="Complete name as per official documents"
                )
                
                email = st.text_input(
                    "Email Address *", 
                    placeholder="candidate@email.com",
                    help="Primary email address for communication"
                )
                
                phone = st.text_input(
                    "WhatsApp Number", 
                    placeholder="+62 812-3456-7890",
                    help="WhatsApp number with country code"
                )
                
                birth_date = st.date_input(
                    "Date of Birth",
                    value=date(1995, 1, 1),
                    help="Used to calculate age automatically"
                )
            
            with col2:
                current_address = st.text_area(
                    "Current Address", 
                    placeholder="Street, City, Province, Country",
                    help="Current residential address"
                )
                
                current_status = st.selectbox(
                    "Current Status",
                    options=[
                        "Available",
                        "Employed - Open to opportunities", 
                        "Unemployed",
                        "Student",
                        "Fresh Graduate",
                        "Career Break"
                    ],
                    help="Current employment/availability status"
                )
            
            # Professional Information Section
            st.markdown("### 💼 Professional Information")
            col3, col4 = st.columns(2)
            
            with col3:
                desired_position = st.text_input(
                    "Desired Position *", 
                    placeholder="e.g., Software Engineer, Data Analyst",
                    help="Target job position or role"
                )
                
                experience = st.text_area(
                    "Work Experience", 
                    placeholder="Describe work experience, format: X tahun at Company Y as Position Z",
                    help="Work experience description (system will extract years automatically)"
                )
                
                education_level = st.selectbox(
                    "Highest Education Level *",
                    options=[
                        "High School",
                        "Diploma (D3)",
                        "Bachelor (S1)", 
                        "Master (S2)",
                        "Doctorate (S3)",
                        "Professional Certification"
                    ],
                    index=2,  # Default to Bachelor
                    help="Highest completed education level"
                )
            
            with col4:
                faculty_major = st.text_input(
                    "Faculty/Major", 
                    placeholder="e.g., Computer Science, Business Administration",
                    help="Field of study or major"
                )
                
                expected_salary = st.number_input(
                    "Expected Salary (IDR) *",
                    min_value=1000000,
                    max_value=100000000,
                    value=8000000,
                    step=500000,
                    help="Expected monthly salary in Indonesian Rupiah"
                )
            
            # Skills Section
            st.markdown("### 🛠️ Skills & Competencies")
            skills_input = st.text_area(
                "Skills *", 
                placeholder="Enter skills separated by commas\nExample: Python, SQL, Machine Learning, Communication Skills, Project Management",
                help="List all relevant skills separated by commas",
                height=120
            )
            
            # Additional Information
            st.markdown("### 📋 Additional Information (Optional)")
            col5, col6 = st.columns(2)
            
            with col5:
                portfolio_url = st.text_input(
                    "Portfolio/LinkedIn URL", 
                    placeholder="https://linkedin.com/in/username",
                    help="Online portfolio or professional profile"
                )
                
                languages = st.text_input(
                    "Languages Spoken", 
                    placeholder="Indonesian (Native), English (Fluent), Mandarin (Basic)",
                    help="Languages and proficiency levels"
                )
            
            with col6:
                certifications = st.text_area(
                    "Certifications", 
                    placeholder="Professional certifications, licenses, or courses completed",
                    help="Relevant professional certifications"
                )
                
                notes = st.text_area(
                    "Additional Notes", 
                    placeholder="Any additional information about the candidate",
                    help="Internal notes or special considerations"
                )
            
            # Form Submission
            col7, col8, col9 = st.columns([1, 2, 1])
            with col8:
                submitted = st.form_submit_button(
                    "➕ Add Candidate", 
                    type="primary",
                    use_container_width=True
                )
        
        # Process Form Submission
        if submitted:
            # Validate required fields
            required_fields = {
                'Full Name': full_name,
                'Email Address': email,
                'Desired Position': desired_position,
                'Skills': skills_input
            }
            
            missing_fields = [field for field, value in required_fields.items() if not value or value.strip() == ""]
            
            if missing_fields:
                st.error(f"❌ Please fill in required fields: {', '.join(missing_fields)}")
            elif not validate_email(email):
                st.error("❌ Please enter a valid email address")
            else:
                # Prepare candidate data
                candidate_data = {
                    'full_name': full_name,
                    'email': email,
                    'phone': phone,
                    'birth_date': birth_date,
                    'current_address': current_address,
                    'current_status': current_status,
                    'desired_position': desired_position,
                    'experience': experience,
                    'education_level': education_level,
                    'faculty_major': faculty_major,
                    'expected_salary': expected_salary,
                    'skills_input': skills_input,
                    'portfolio_url': portfolio_url,
                    'languages': languages,
                    'certifications': certifications,
                    'notes': notes
                }
                
                # Add candidate to system
                success, message = add_candidate_to_system(cv_system, candidate_data)
                
                if success:
                    st.success("✅ " + message)
                    st.balloons()
                    
                    # Show added candidate summary
                    with st.expander("📋 Added Candidate Summary", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Name:** {full_name}")
                            st.write(f"**Email:** {email}")
                            st.write(f"**Position:** {desired_position}")
                            st.write(f"**Education:** {education_level}")
                        with col2:
                            st.write(f"**Expected Salary:** {format_salary(expected_salary)}")
                            skills_list = process_skills_input(skills_input)
                            st.write(f"**Skills Count:** {len(skills_list)}")
                            st.write(f"**Status:** {current_status}")
                            age = calculate_age_from_date(birth_date)
                            st.write(f"**Age:** {age} years")
                    
                    # Option to immediately get recommendations
                    st.info("💡 You can now find this candidate in recommendations for relevant job positions!")
                    
                    # Update session state to reflect new candidate count
                    st.rerun()
                else:
                    st.error("❌ " + message)
        
        # Recent Additions
        if len(cv_system.candidates_data) > 0:
            st.subheader("📊 Recent Additions")
            
            # Show last 5 candidates added (assuming newer candidates have higher indices)
            recent_candidates = cv_system.candidates_data.tail(5)[
                ['Full name', 'Desired positions', 'Highest formal of education', 'skills_count', 'expected_salary_normalized']
            ].copy()
            
            recent_candidates = recent_candidates.rename(columns={
                'Full name': 'Name',
                'Desired positions': 'Position',
                'Highest formal of education': 'Education',
                'skills_count': 'Skills',
                'expected_salary_normalized': 'Expected Salary'
            })
            
            # Format salary
            recent_candidates['Expected Salary'] = recent_candidates['Expected Salary'].apply(format_salary)
            
            st.dataframe(recent_candidates, use_container_width=True)
        
        # Quick Import Section
        st.subheader("📤 Quick Import from CSV")
        st.write("Upload a CSV file with candidate data for bulk import.")
        
        uploaded_csv = st.file_uploader(
            "Upload CSV File", 
            type=['csv'],
            help="CSV should contain columns: full_name, email, desired_position, skills (comma-separated)"
        )
        
        if uploaded_csv:
            try:
                import_df = pd.read_csv(uploaded_csv)
                st.write("📋 Preview of uploaded data:")
                st.dataframe(import_df.head(), use_container_width=True)
                
                if st.button("📤 Import Candidates from CSV"):
                    success_count = 0
                    error_count = 0
                    
                    for index, row in import_df.iterrows():
                        try:
                            candidate_data = {
                                'full_name': str(row.get('full_name', row.get('name', ''))),
                                'email': str(row.get('email', '')),
                                'phone': str(row.get('phone', '')),
                                'birth_date': date(1995, 1, 1),  # Default date
                                'current_address': str(row.get('address', 'Not specified')),
                                'current_status': str(row.get('status', 'Available')),
                                'desired_position': str(row.get('desired_position', row.get('position', ''))),
                                'experience': str(row.get('experience', 'Fresh Graduate')),
                                'education_level': str(row.get('education', 'Bachelor')),
                                'faculty_major': str(row.get('major', 'General')),
                                'expected_salary': int(row.get('salary', 8000000)),
                                'skills_input': str(row.get('skills', '')),
                                'portfolio_url': '',
                                'languages': '',
                                'certifications': '',
                                'notes': f'Imported on {datetime.now().strftime("%Y-%m-%d %H:%M")}'
                            }
                            
                            # Validate required fields
                            if (candidate_data['full_name'] and 
                                candidate_data['email'] and 
                                candidate_data['desired_position'] and 
                                candidate_data['skills_input'] and
                                validate_email(candidate_data['email'])):
                                
                                success, message = add_candidate_to_system(cv_system, candidate_data)
                                if success:
                                    success_count += 1
                                else:
                                    error_count += 1
                            else:
                                error_count += 1
                                
                        except Exception as e:
                            error_count += 1
                            continue
                    
                    if success_count > 0:
                        st.success(f"✅ Successfully imported {success_count} candidates!")
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} candidates failed to import (missing data, duplicates, or errors)")
                    
                    if success_count > 0:
                        st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")

    # =============================================================================
    # TAB 5: JOB PROFILES
    # =============================================================================
    with tab5:
        st.header("💼 Job Profiles Management")
        
        # Current Job Profiles
        st.subheader("📋 Current Job Profiles")
        
        for job_id, profile in cv_system.job_profiles.items():
            with st.expander(f"📝 {profile['title']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Job Details:**")
                    st.write(f"Description: {profile['description']}")
                    st.write(f"Experience: {profile['min_experience']}-{profile['max_experience']} years")
                    st.write(f"Education: {profile['preferred_education']}")
                    
                    salary_min, salary_max = profile['salary_range']
                    st.write(f"Salary: {format_salary(salary_min)} - {format_salary(salary_max)}")
                    
                    st.write("**Required Skills:**")
                    for skill in profile['required_skills']:
                        st.write(f"• {skill}")
                
                with col2:
                    st.write("**Preferred Skills:**")
                    for skill in profile['preferred_skills']:
                        st.write(f"• {skill}")
                    
                    st.write("**Keywords:**")
                    for keyword in profile['keywords']:
                        st.write(f"• {keyword}")
        
        # Add New Job Profile
        st.subheader("➕ Add New Job Profile")
        
        with st.expander("Create New Job Profile", expanded=False):
            with st.form("new_job_profile"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_job_id = st.text_input("Job ID (unique identifier):")
                    new_job_title = st.text_input("Job Title:")
                    new_job_description = st.text_area("Job Description:")
                    
                    new_min_exp = st.number_input("Minimum Experience (years):", 0, 20, 0)
                    new_max_exp = st.number_input("Maximum Experience (years):", 0, 20, 5)
                    
                    new_education = st.selectbox("Preferred Education:", 
                                               ["High School", "Diploma", "Bachelor", "Master", "PhD"])
                
                with col2:
                    new_required_skills = st.text_area(
                        "Required Skills (one per line):",
                        help="Enter each skill on a new line"
                    )
                    
                    new_preferred_skills = st.text_area(
                        "Preferred Skills (one per line):",
                        help="Enter each skill on a new line"
                    )
                    
                    new_salary_min = st.number_input("Minimum Salary (IDR):", 0, 100000000, 5000000)
                    new_salary_max = st.number_input("Maximum Salary (IDR):", 0, 100000000, 15000000)
                
                submitted = st.form_submit_button("Create Job Profile")
                
                if submitted and new_job_id and new_job_title:
                    required_skills_list = [skill.strip() for skill in new_required_skills.split('\n') if skill.strip()]
                    preferred_skills_list = [skill.strip() for skill in new_preferred_skills.split('\n') if skill.strip()]
                    
                    new_profile = {
                        'title': new_job_title,
                        'description': new_job_description,
                        'required_skills': required_skills_list,
                        'preferred_skills': preferred_skills_list,
                        'min_experience': new_min_exp,
                        'max_experience': new_max_exp,
                        'preferred_education': new_education,
                        'salary_range': (new_salary_min, new_salary_max),
                        'keywords': [new_job_title.lower(), 'job', 'position']
                    }
                    
                    # Process for TF-IDF
                    job_text = f"{new_job_title} {new_job_description}"
                    skills_text = ' '.join(required_skills_list + preferred_skills_list)
                    keywords_text = ' '.join(new_profile['keywords'])
                    full_job_text = f"{job_text} {skills_text} {skills_text} {keywords_text}"
                    
                    new_profile['processed_skills_text'] = cv_system.preprocess_text(skills_text)
                    new_profile['processed_full_text'] = cv_system.preprocess_text(full_job_text)
                    
                    # Add to system
                    cv_system.job_profiles[new_job_id] = new_profile
                    
                    st.success(f"✅ Job profile '{new_job_title}' created successfully!")
                    st.rerun()
    
    # =============================================================================
    # TAB 6: SETTINGS
    # =============================================================================
    with tab6:
        st.header("⚙️ System Settings")
        
        # Model Information
        st.subheader("📊 Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**System Details:**")
            st.write(f"Model Version: {cv_system.model_version}")
            st.write(f"Training Date: {cv_system.training_date.strftime('%Y-%m-%d %H:%M:%S') if cv_system.training_date else 'Unknown'}")
            st.write(f"Total Candidates: {len(cv_system.candidates_data)}")
            st.write(f"Total Job Profiles: {len(cv_system.job_profiles)}")
        
        with col2:
            st.write("**Model Components:**")
            st.write(f"Skills Vocabulary Size: {len(cv_system.skills_tfidf_vectorizer.vocabulary_)}")
            st.write(f"Content Vocabulary Size: {len(cv_system.content_tfidf_vectorizer.vocabulary_)}")
            if cv_system.svd_model:
                st.write(f"SVD Components: {cv_system.svd_model.n_components}")
                st.write(f"Explained Variance: {cv_system.svd_model.explained_variance_ratio_.sum():.4f}")
        
        # Scoring Weights Configuration
        st.subheader("⚖️ Scoring Weights Configuration")
        st.write("Adjust the weights for different scoring components:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            skills_weight = st.slider("Skills Matching Weight:", 0.0, 1.0, 0.40, 0.05)
        with col2:
            content_weight = st.slider("Content-Based Weight:", 0.0, 1.0, 0.25, 0.05)
        with col3:
            collab_weight = st.slider("Collaborative Weight:", 0.0, 1.0, 0.20, 0.05)
        with col4:
            feature_weight = st.slider("Feature-Based Weight:", 0.0, 1.0, 0.15, 0.05)
        
        total_weight = skills_weight + content_weight + collab_weight + feature_weight
        
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights should sum to 1.0 (current: {total_weight:.2f})")
        else:
            st.success("✅ Weights are properly normalized!")
        
        # Model Management
        st.subheader("💾 Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Current Model"):
                filename = f"cv_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                try:
                    model_data = {
                        'model_version': cv_system.model_version,
                        'training_date': cv_system.training_date,
                        'candidates_data': cv_system.candidates_data,
                        'skills_tfidf_vectorizer': cv_system.skills_tfidf_vectorizer,
                        'skills_tfidf_matrix': cv_system.skills_tfidf_matrix,
                        'content_tfidf_vectorizer': cv_system.content_tfidf_vectorizer,
                        'content_tfidf_matrix': cv_system.content_tfidf_matrix,
                        'user_item_matrix': cv_system.user_item_matrix,
                        'svd_model': cv_system.svd_model,
                        'user_factors': cv_system.user_factors,
                        'item_factors': cv_system.item_factors,
                        'predicted_interactions': cv_system.predicted_interactions,
                        'job_profiles': cv_system.job_profiles
                    }
                    
                    with open(filename, 'wb') as f:
                        pickle.dump(model_data, f)
                    
                    st.success(f"✅ Model saved as {filename}")
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
        
        with col2:
            if st.button("🔄 Reset Session"):
                st.session_state.cv_system = None
                st.session_state.recommendations = {}
                st.success("✅ Session reset successfully!")
                st.rerun()
        
        # Database Export
        st.subheader("📥 Database Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Candidates CSV"):
                csv_data = cv_system.candidates_data.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Export Candidates JSON"):
                json_data = cv_system.candidates_data.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download JSON",
                    data=json_data,
                    file_name=f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📈 Export to Excel"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    cv_system.candidates_data.to_excel(writer, sheet_name='Candidates', index=False)
                    
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Candidates', 'Avg Skills Count', 'Avg Experience', 'Avg Salary'],
                        'Value': [
                            len(cv_system.candidates_data),
                            cv_system.candidates_data['skills_count'].mean(),
                            cv_system.candidates_data['experience_years'].mean(),
                            cv_system.candidates_data['expected_salary_normalized'].mean()
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📈 Download Excel",
                    data=output.getvalue(),
                    file_name=f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # System Statistics
        st.subheader("📈 System Statistics")
        
        if st.session_state.recommendations:
            total_recs = sum(len(recs) for recs in st.session_state.recommendations.values())
            total_jobs = len(st.session_state.recommendations)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Recommendations Generated", total_recs)
            with col2:
                st.metric("Jobs Processed", total_jobs)
            with col3:
                avg_recs_per_job = total_recs / total_jobs if total_jobs > 0 else 0
                st.metric("Avg Recommendations per Job", f"{avg_recs_per_job:.1f}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_recommendation_report(job_profile, recommendations):
    """Generate a detailed text report"""
    report = f"""
CV RECOMMENDATION REPORT
========================

Job Position: {job_profile['title']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

JOB REQUIREMENTS:
-----------------
Description: {job_profile['description']}
Experience: {job_profile['min_experience']}-{job_profile['max_experience']} years
Education: {job_profile['preferred_education']}
Salary Range: {format_salary(job_profile['salary_range'][0])} - {format_salary(job_profile['salary_range'][1])}

Required Skills: {', '.join(job_profile['required_skills'])}
Preferred Skills: {', '.join(job_profile['preferred_skills'])}

RECOMMENDATIONS:
================

"""
    
    for i, rec in enumerate(recommendations, 1):
        report += f"""
CANDIDATE #{i}: {rec['name']}
{'='*50}
Email: {rec['email']}
Phone: {rec['phone']}
Education: {rec['education']} - {rec['major']}
Experience: {rec['experience_years']} years
Expected Salary: {format_salary(rec['expected_salary'])}
Location: {rec['address']}

SCORING BREAKDOWN:
- Final Score: {rec['final_score']}%
- Skills Match: {rec['skills_match_score']}%
- Content-Based: {rec['content_similarity_score']}%
- Collaborative: {rec['collaborative_score']}%
- Features: {rec['feature_score']}%

MATCHED SKILLS:
Required: {', '.join(rec['matched_required_skills']) if rec['matched_required_skills'] else 'None'}
Preferred: {', '.join(rec['matched_preferred_skills']) if rec['matched_preferred_skills'] else 'None'}

"""
    
    # Summary statistics
    avg_score = np.mean([r['final_score'] for r in recommendations])
    top_score = max([r['final_score'] for r in recommendations])
    avg_skills_match = np.mean([len(r['matched_required_skills']) for r in recommendations])
    
    report += f"""
SUMMARY STATISTICS:
===================
Total Candidates Evaluated: {len(recommendations)}
Average Final Score: {avg_score:.1f}%
Top Score: {top_score:.1f}%
Average Matched Required Skills: {avg_skills_match:.1f}

Generated by CV Recommendation System v1.0
"""
    
    return report

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()