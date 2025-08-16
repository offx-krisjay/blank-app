import streamlit as st
import pandas as pd 
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Data-Cleansing, Profiling & ML Tool")
st.subheader("By Muralikrishna")

up_file = st.file_uploader("Upload CSV", type=['csv'])
if up_file is not None:
    pl_file = pl.read_csv(up_file)
    st.toast("Data Loaded successfully")
    st.expander("Top 10 Rows").write(pl_file.head(10))
    st.expander("Bottom 10 Rows").write(pl_file.tail(10))

    df=pl_file.to_pandas()
    
    st.sidebar.header("Data Cleaning Operations")
    if st.sidebar.checkbox("Drop NaN values"):
        df.dropna(inplace=True)
    if st.sidebar.checkbox("Remove Duplicate rows"):
        df.drop_duplicates(inplace=True)
    if st.sidebar.checkbox("Normalize numeric columns"):
        scaler = MinMaxScaler()
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = scaler.fit_transform(df[num_cols])
    if st.sidebar.checkbox("Type Conversion"):
        col_conv=st.sidebar.selectbox("Select column to convert",df.columns)
        new_dtype=st.sidebar.selectbox("select new type",["int","float"])
        try:
            df[col_conv]=df[col_conv].astype(new_dtype)
        except Exception as e:
            st.error(f"Conversion caused an error:{e}")
    if st.sidebar.checkbox("filter rows"):
        filt_col=st.sidebar.selectbox("Select column to filter", df.columns)
        filt_val=st.sidebar.text_input("enter filter value")
        if filt_val:
            df=df[df[filt_col].astype(str).str.contains(filt_val)]
        
    st.subheader("Cleaned Data Preview")
    if df.empty:
        st.warning("DataFrame is empty after cleaning.")
    else:
        st.dataframe(df.head(100))

    st.subheader("Profiling Report")
    if st.checkbox("Generate profiling report"):
        if df.empty or df.shape[1] == 0:
            st.warning("DataFrame is empty. Cannot generate profiling report.")
        else:
            st.write("## 1. Column-wise Visualizations")
            for col in df.columns:
                st.subheader(f"📌 {col}")
                fig, ax = plt.subplots()

                if df[col].dtype in ["int64", "float64"]:
                    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
                    plt.title(f"Histogram of {col}")
                else:
                    df[col].value_counts().plot(kind="bar", ax=ax)
                    plt.title(f"Bar Chart of {col}")

                st.pyplot(fig)

            st.write("## 2. Correlation Heatmap")
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            plt.title("Correlation Heatmap")
            st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

 
    st.subheader("Encoding")

    if st.checkbox("Convert all String column to numerical value"):
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        enc_type = st.selectbox("Encoding type", ["Label Encoding","One-Hot Encoding"])
        if st.button("Encode"): 
            if enc_type == "Label Encoding":
                le = LabelEncoder()
                for col in cat_cols:
                    df[col + "_enc"] = le.fit_transform(df[col])
                df.drop(columns=cat_cols, inplace=True)
            else: 
                df = pd.get_dummies(df, columns=cat_cols)
                df.drop(columns=cat_cols, inplace=True)
    st.subheader("Encoded Data Preview")
    st.dataframe(df.head(100))
        
    
 
    st.subheader("Machine Learning")
    ml_type=st.selectbox("Type",["Regression","Classification"])
    features=st.multiselect("Select Features",df.columns)
    target=st.selectbox("Select target",df.columns)

    if features and target:
        x,y = df[features],df[target]
        xtrain, xtest, ytrain,ytest=train_test_split(x,y,test_size=0.3, random_state=42)

        if ml_type == "Regression":
            model = RandomForestRegressor()
            model.fit(xtrain,ytrain)
            rpred=model.predict(xtest)
            st.write("R2 Score:",r2_score(ytest,rpred))
            st.write("Mean squared error:",mean_squared_error(ytest,rpred))
        else:
            model = RandomForestClassifier()
            model.fit(xtrain,ytrain)
            cpred=model.predict(xtest)
            st.write("Accuracy:",accuracy_score(ytest,cpred)) # type: ignore
            st.text(classification_report(ytest,cpred))
        
    if ml_type is not None:
        st.subheader("User Input prediction")
        ip_pred=st.radio("Predict using:",["Manual Input","Upload File"])
        if ip_pred == "Manual Input":
            input_df={}
            for filt_col in features:
                filt_val=st.text_input(f"Value for {filt_col}")
                input_df[filt_col]=float(filt_val) if filt_val else 0
            if st.button("Predict"):
                inp_pred=model.predict(pd.DataFrame([input_df]))
                st.success(f"Prediction: {pred[0]}")
        else:
            us_file=st.file_uploader("Upload new csv for prediction",type=["csv"])
            if us_file:
                new_df = pd.read_csv(us_file)
                us_pred=model.predict(us_df[features]) # type: ignore
                us_df['Predicitions']=us_pred
                st.dataframe(us_df)
                st.download_button("download predictions for user file",us_df.to_csv(Index=False),"user_file_predictions.csv")
