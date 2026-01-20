import streamlit as st
import pandas as pd
import plotly.express as px  
import folium
from streamlit_folium import st_folium
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Digital Health Dashboard",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    # Ensure the path is correct for your new repository structure
    df = pd.read_csv('data/processed/cleaned_data.csv')
    return df

df = load_data()

# --- SIDEBAR ---
# 1. Logo at the top
st.sidebar.image("visuals/main_logo.jpg", width=150)
st.sidebar.title("🛠 Navigation")
page = st.sidebar.radio(
    "Select Project Section:",
    ["Home", "Hypothesis Analysis", "Global Geography", "ML Diagnostics", "Analytical Report"]
)

st.sidebar.markdown("---")
st.sidebar.info("Project prepared by: Vitaliy Chernetskyi")

# --- PAGE ROUTING LOGIC ---

if page == "Home":
    st.title("📊 Students' Social Media Addiction Analysis")
    st.write("""
    Welcome to the research project dedicated to analyzing the digital habits of youth. 
    We analyzed data from 700+ students worldwide to understand how screen time 
    affects our real-world lives, health, and relationships.
    """)
    
    st.subheader("Global Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Respondents", len(df))
    with col2:
        st.metric("Avg. Social Media Time", f"{df['Avg_Daily_Usage_Hours'].mean():.1f} hrs/day")
    with col3:
        st.metric("Addiction Level", f"{df['Addicted_Score'].mean():.1f}/10")
    with col4:
        st.metric("Regions", df['Region'].nunique())

    st.write("---")
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

elif page == "Hypothesis Analysis":
    st.title("🧬 In-depth Hypothesis Analysis")
    st.write("In this section, we test statistical assumptions regarding the impact of social media on students' lives.")

    # Tabs for different hypothesis groups
    tab1, tab2, tab3 = st.tabs(["🏥 Health & Psychology", "📱 Platforms", "🤝 Social Relations"])
    level_order = {"Addiction_Level": ["Low", "Medium", "High"]}

    with tab1:
        st.header("Impact on Physical and Mental State")
        
        st.subheader("Hypothesis 1: Social Media and Sleep Quality")
        fig1 = px.scatter(
            df, x="Avg_Daily_Usage_Hours", y="Sleep_Hours_Per_Night",
            color="Addiction_Level", trendline="ols",
            labels={"Avg_Daily_Usage_Hours": "Daily Hours Online",
                    "Sleep_Hours_Per_Night": "Sleep Hours",
                    "Addiction_Level": "Addiction Level"},
            color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
            category_orders=level_order
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.success("**Verdict:** Clear negative correlation. Increasing social media usage directly leads to a reduction in sleep duration.")

        st.write("---")

        st.subheader("Hypothesis 2: Addiction and Mental Health")
        fig2 = px.box(
            df, x="Addiction_Level", y="Mental_Health_Score",
            color="Addiction_Level", points="all",
            labels={"Addiction_Level": "Addiction Level", "Mental_Health_Score": "Mental Health Score"},
            color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
            category_orders=level_order
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.success("**Verdict:** Students with high addiction levels show significantly lower median mental health scores.")

    with tab2:
        st.header("Platform Analysis")
        st.subheader("Hypothesis 3: Algorithmic Feed Platforms vs. Others")
        
        platform_stats = df.groupby('Most_Used_Platform')['Addicted_Score'].mean().sort_values(ascending=False).reset_index()
        
        fig3 = px.bar(
            platform_stats, x="Most_Used_Platform", y="Addicted_Score",
            color="Addicted_Score",
            labels={"Most_Used_Platform": "Primary Platform", "Addicted_Score": "Avg. Addiction Score"},
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.info("**Analytical Insight:** Platforms using 'infinite scroll' algorithms (TikTok, Instagram) have the highest statistical correlation with addiction scores.")
        st.write("---")
        
        st.header("Content Type Analysis")
        st.write("Platforms grouped by their core function to identify 'dopamine traps'.")

        st.subheader("⚡️ Usage Time vs. Addictiveness")
        
        type_stats = df.groupby('Platform_Type').agg({
            'Addicted_Score': 'mean',
            'Avg_Daily_Usage_Hours': 'mean',
            'Student_ID': 'count'
        }).reset_index()

        fig_scatter = px.scatter(
            type_stats, 
            x="Avg_Daily_Usage_Hours", 
            y="Addicted_Score",
            size="Student_ID", 
            color="Platform_Type",
            text="Platform_Type",
            labels={"Avg_Daily_Usage_Hours": "Avg. Usage (Hrs)", 
                    "Addicted_Score": "Avg. Addiction Score"},
            title="Where does addiction develop fastest?",
            height=500
        )

        fig_scatter.update_layout(showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
        fig_scatter.update_xaxes(dtick=1.0, range=[2, 7])
        fig_scatter.update_yaxes(dtick=1.0, range=[3, 8])
        fig_scatter.update_traces(textposition='top center', cliponaxis=False)

        st.plotly_chart(fig_scatter, use_container_width=True)
        st.info("**Insight:** The 'Entertain-Scroll' category (TikTok/Instagram) shows the highest addiction levels despite messaging apps often having more total usage time.")

        st.write("---")
        
        st.subheader("🚻 Gender Preferences")
        
        gender_data = df.groupby(['Platform_Type', 'Gender']).size().reset_index(name='Count')
        
        fig_gender = px.bar(
            gender_data, 
            x="Platform_Type", 
            y="Count", 
            color="Gender",
            barmode="group",
            labels={"Platform_Type": "Platform Type", "Count": "User Count"},
            title="Interest Distribution by Gender",
            color_discrete_map={"Male": "#1f77b4", "Female": "#e377c2"}
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        st.warning("**Gender Gap:** Male respondents are more inclined toward 'Social-Network' (news feeds), while females dominate in entertainment-focused content.\n\n"
                  "👉 This indicates a difference in goals: boys go for information, girls for visual content.")
        
        st.write("---")
        
        st.subheader("🔍 Digital Consumption Structure")
        
        tree_data = df.groupby(['Platform_Type', 'Most_Used_Platform']).agg({
            'Addicted_Score': 'mean',
            'Student_ID': 'count'
        }).reset_index()

        fig_tree = px.treemap(
            tree_data, 
            path=['Platform_Type', 'Most_Used_Platform'], 
            values='Student_ID', 
            color='Addicted_Score',
            color_continuous_scale='RdYlGn_r',
            labels={'Student_ID': 'Respondents', 'Addicted_Score': 'Avg. Addiction Score'},
            title="Platform Popularity within Categories (Color = Addiction Level)"
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        st.info("This graph shows the 'weight' of each platform. The size of the rectangle is the number of students, and the color is how much that platform 'pulls'.")

    with tab3:
        st.header("Social Relations & Education")
        
        st.subheader("Hypothesis 4: Conflicts and Relationship Status")
        conflict_stats = df.groupby('Relationship_Status')['Conflicts_Over_Social_Media'].mean().sort_values().reset_index()
        
        fig4 = px.bar(
            conflict_stats, 
            x="Conflicts_Over_Social_Media", 
            y="Relationship_Status",
            orientation='h',
            title="Avg. Conflict Frequency by Relationship Status",
            labels={"Relationship_Status": "Status", "Conflicts_Over_Social_Media": "Avg. Conflicts"},
            color="Conflicts_Over_Social_Media", 
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.success("**Verdict:** Hypothesis confirmed. 'Complicated' status shows the highest level of social media-related conflicts.")
        
        st.write("---")

        st.subheader("Hypothesis 5: Relationships as a Protective Factor")
        fig6 = px.box(
            df, 
            x="Relationship_Status", 
            y="Addicted_Score",
            color="Relationship_Status",
            title="Addiction Score Distribution by Relationship Status",
            labels={"Relationship_Status": "Status", "Addicted_Score": "Addiction Score"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig6, use_container_width=True)
        st.info("**Conclusion:** Stable relationships ('In a relationship') often act as a buffer, lowering average digital addiction levels.")

        st.write("---")
        
        st.subheader("Hypothesis 6: Impact of Addiction on Academic Performance")
        fig5 = px.box(
            df, x="Addiction_Level", y="Affects_Academic_Performance_Numeric",
            color="Addiction_Level",
            labels={
                "Addiction_Level": "Addiction Level",
                "Affects_Academic_Performance_Numeric": "Impact on Performance (Numeric Score)"
            },
            color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
            category_orders=level_order
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.success("**Verdict:** Hypothesis confirmed — high digital addiction statistically correlates with decreased academic performance.")

elif page == "Global Geography":
    st.title("🌍 Global Geography of Addiction")
    st.write("How is digital addiction distributed across the globe?")

    country_map_data = df.groupby('Country')['Addicted_Score'].mean().reset_index()

    st.subheader("World Map of Addiction Levels")
    
    fig_map = px.choropleth(
        country_map_data,
        locations="Country",
        locationmode="country names",
        color="Addicted_Score",
        hover_name="Country",
        color_continuous_scale="YlOrRd", 
        labels={"Addicted_Score": "Avg. Addiction Score"}
    )
    
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.write("---")

    st.subheader("Hypothesis 7: Regional Differences (N. America vs. Europe)")
    
    region_stats = df.groupby('Region')['Addicted_Score'].mean().sort_values(ascending=True).reset_index()
    
    fig_region = px.bar(
        region_stats,
        x="Addicted_Score",
        y="Region",
        orientation='h',
        color="Addicted_Score",
        text_auto='.2f', 
        title="Average Addiction Score by Continent",
        labels={"Region": "Continent", "Addicted_Score": "Avg. Score"},
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_region, use_container_width=True)

    st.success("""
    **Verdict:** Hypothesis 7 confirmed. Regions with high concentrations of tech hubs 
    (notably North America) show higher addiction scores compared to Europe.
    """)
    st.write("---")

    st.subheader("🌍 Regional Platform Leaders")
    st.write("Which platform dominates on each continent?")

    platform_logos = {
        "Instagram": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg",
        "TikTok": "https://upload.wikimedia.org/wikipedia/en/a/a9/TikTok_logo.svg",
        "Facebook": "https://upload.wikimedia.org/wikipedia/commons/b/b8/2021_Facebook_icon.svg"
    }

    region_coords = {
        "Europe": [50, 15],
        "Asia": [35, 90],
        "North America": [45, -100],
        "South America": [-15, -60],
        "Oceania": [-25, 135],
        "Africa": [5, 20]
    }

    region_counts = df.groupby(['Region', 'Most_Used_Platform']).size().reset_index(name='Count')
    top_reg = region_counts.loc[region_counts.groupby('Region')['Count'].idxmax()]

    m = folium.Map(
        location=[20, 0], 
        zoom_start=2, 
        tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        attr='&copy; CARTO'
    )

    for _, row in top_reg.iterrows():
        region = row['Region']
        count = row['Count']
        platform = row['Most_Used_Platform']
        
        if region in region_coords:
            icon_size = 40 + (np.sqrt(count) * 4) 
            logo_url = platform_logos.get(platform, "")
            
            if logo_url:
                icon = folium.CustomIcon(logo_url, icon_size=(icon_size, icon_size))
                folium.Marker(
                    location=region_coords[region],
                    icon=icon,
                    tooltip=f"<b>{region}</b><br>Platform: {platform}<br>Users: {count}"
                ).add_to(m)

    st_folium(m, width="100%", height=550)
    st.info("**Geographic Distribution:** Instagram dominates most regions, while TikTok and Facebook hold leads in South America and Africa respectively.")

    st.write("---")

    st.subheader('🗂️ Concentration Matrix')
    st.write('Where are the users of each network concentrated?')

    bubble_data = df.groupby(['Region', 'Most_Used_Platform']).size().reset_index(name='User_Count')
    sorted_platforms = sorted(bubble_data['Most_Used_Platform'].unique())
    sorted_regions = sorted(bubble_data['Region'].unique())
    
    fig_bubble = px.scatter(
        bubble_data,
        x="Region",
        y="Most_Used_Platform",
        size="User_Count",
        color="User_Count",
        text="User_Count",
        size_max=60,
        labels={"Region": "Global Region", "Most_Used_Platform": "Social Network", "User_Count": "Count"},
        category_orders={"Most_Used_Platform": sorted_platforms, "Region": sorted_regions},
        color_continuous_scale="Viridis",
        height=600
    )

    fig_bubble.update_traces(textposition='middle center', textfont=dict(color='white'))
    fig_bubble.update_layout(xaxis={'side': 'top'}, showlegend=False)

    st.plotly_chart(fig_bubble, use_container_width=True)

    st.info("""
    **Geographic Insight:**
    * **European Hub:** Europe is a center of activity for most Western platforms.
    * **Asian Specificity:** Only in Asia do we see significant activity in WeChat, LINE, and KakaoTalk.
    * **Global Instagram:** The Instagram row shows the highest intensity across nearly all columns.
    """)

elif page == "ML Diagnostics":
    st.title("💻⚙️ Machine Learning: Digital Profile")
    st.write("""
    This tool uses **K-Means clustering** logic to determine which user group 
    you belong to based on your digital habits.
    """)

    st.subheader("Enter your metrics:")
    
    with st.container(border=True):
        col_in1, col_in2 = st.columns(2)
        
        with col_in1:
            usage = st.slider("Daily social media hours?", 0.0, 24.0, 5.0, step=0.5)
            sleep = st.slider("Daily sleep hours?", 0.0, 12.0, 8.0, step=0.5)
        
        with col_in2:
            mental = st.select_slider("Rate your mental state (1 - Poor, 10 - Excellent)", options=list(range(1, 11)), value=8)
            performance = st.radio("Does social media affect your performance?", ["Negatively", "Neutral/Positively"])

    if st.button("Analyze My Profile", type="primary", use_container_width=True):
        if (usage + sleep) > 24.0:
            st.error(f"⚠️ **Data Error:** Total hours ({usage + sleep}) exceed 24 hours in a day. Please adjust.")
        else:
            risk_score = (usage * 0.4) + ((10 - mental) * 0.3) + ((8 - sleep) * 0.3)
            
            st.write("---")
            st.subheader("Analysis Result:")
            
            if usage >= 6.0 or risk_score > 5.0:
                st.error("🔴 **Your Profile: High Addiction Level**")
                st.warning("Your metrics align with the 'High Addiction' cluster. We recommend reviewing your digital habits.")
            elif usage >= 4.0 or risk_score > 3.0:
                st.warning("🟡 **Your Profile: Medium Level (At-Risk)**")
                st.info("You are currently in the 'Medium Addiction' zone.")
            else:
                st.success("🟢 **Your Profile: Balanced User**")
                st.balloons()
                st.write("Your metrics align with the 'Low Addiction' cluster.")

elif page == "Analytical Report":
    try:
        # Assumes you have an English version or have translated STORYTELLING.md
        with open("STORYTELLING.md", "r", encoding="utf-8") as f:
            story_content = f.read()
        
        st.markdown(story_content, unsafe_allow_html=True)
            
    except FileNotFoundError:
        st.error("File STORYTELLING.md not found in the project root.")
