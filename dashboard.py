import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yt_comment_extractor as yt 

st.set_page_config(layout="wide")

st.header(":blue-background[Sentiment Analyzer for your brand campaign]")




choice = st.selectbox("Select Platform from given list",["Youtube","Instagram","Twitter"])

if choice == "Youtube":
    method_choice = st.selectbox("Select a method",["Selenium"," Youtube API "])

    if method_choice == "Selenium":

        video_url = st.text_input("### Enter Youtube Video link:")
        total_comments = int(st.number_input("### Enter number of comments you want to analyze"))

        analyze = st.button("Analyze")
        
        if analyze:
            yt.yt_run(video_url,total_comments)

            chart_data = pd.read_csv("data.csv")

            st.subheader("Pie Chart")
            
            plt.pie(chart_data['Count'], labels=chart_data['Emotion'])
            st.pyplot( plt )

            st.bar_chart(chart_data,x="Emotion",y="Count")

    else:
        st.write("## Coming soon....")
    

else:
    st.write("### Coming soon...")

