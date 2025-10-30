import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from prophet import Prophet
from datetime import datetime

# ----------------- Streamlit App Setup -----------------
st.set_page_config(page_title="Customer Feedback AI", layout="wide")
st.title("Intelligent Customer Feedback Analysis System")
st.write("This application analyzes customer feedback to understand sentiment, generate summaries, and predict satisfaction trends using AI models.")

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("Upload your customer feedback (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
        st.dataframe(data.head())

        if "feedback_text" not in data.columns:
            st.error("The uploaded file must contain a 'feedback_text' column.")
        else:
            # ----------------- Sentiment Analysis -----------------
            st.subheader("Sentiment Analysis")

            sentiment_task = pipeline("sentiment-analysis")
            if "sentiment" not in data.columns:
                sentiment_list = []
                for text in data["feedback_text"]:
                    try:
                        label = sentiment_task(text[:512])[0]["label"].lower()
                    except Exception:
                        label = "neutral"
                    sentiment_list.append(label)
                data["sentiment"] = sentiment_list

            st.write(data[["feedback_text", "sentiment"]].head())

            # ----------------- Sentiment Distribution Chart -----------------
            st.subheader("Sentiment Distribution Overview")
            fig, ax = plt.subplots()
            sns.countplot(x="sentiment", data=data, ax=ax)
            ax.set_title("Sentiment Count per Category")
            st.pyplot(fig)

            # ----------------- Text Summarization -----------------
            st.subheader("Feedback Summarization")

            summarize_task = pipeline("summarization", model="facebook/bart-large-cnn")
            summaries = []
            for idx, text in enumerate(data["feedback_text"].head(15)):  # summarize first few to save time
                try:
                    summary = summarize_task(text[:700], max_length=60, min_length=25, do_sample=False)[0]["summary_text"]
                except Exception:
                    summary = "Summary unavailable."
                summaries.append(summary)
            data["summary"] = summaries + [""] * (len(data) - len(summaries))

            st.dataframe(data[["feedback_text", "summary"]].head())

            # ----------------- Sentiment Trend Forecast -----------------
            if "date" in data.columns:
                st.subheader("Satisfaction Trend Prediction")

                try:
                    data["date"] = pd.to_datetime(data["date"], errors="coerce")
                    data = data.dropna(subset=["date"])

                    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
                    data["score"] = data["sentiment"].map(sentiment_map)

                    trend_df = data.groupby("date")["score"].mean().reset_index()
                    trend_df.rename(columns={"date": "ds", "score": "y"}, inplace=True)

                    model = Prophet()
                    model.fit(trend_df)

                    future_dates = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future_dates)

                    fig2 = model.plot(forecast)
                    st.pyplot(fig2)
                except Exception as e:
                    st.warning(f"Trend analysis could not be completed: {e}")
            else:
                st.info("No 'date' column found — skipping trend prediction.")

            # ----------------- Download Option -----------------
            st.subheader("Download Processed Data")
            csv_output = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV File",
                data=csv_output,
                file_name=f"processed_feedback_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            st.success("Analysis complete!")

    except Exception as e:
        st.error(f"Error while processing the file: {e}")
else:
    st.info("Upload a CSV file containing feedback data to begin.")
