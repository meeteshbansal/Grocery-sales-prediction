import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def eda(dataframe):
    st.title("Genral Information About the Data")
    st.write("# Data Overview")

    st.write("### Descrptive Statistics")
    st.write(dataframe.describe())

    fig ,ax = plt.subplots(figsize = (10,6))
    sns.barplot(x = 'City',y = 'Gender',width = 0.2,data = dataframe,errorbar=None,ax = ax)
    ax.set_title('Catgorization of Gender Accoding to City')
    ax.set_xlabel('City')
    ax.set_ylabel('Gender')
    ax.set_xticklabels(ax.get_xticklabels(),rotation =45)
    st.pyplot(fig)

    fig,ax = plt.subplots(figsize = (10,6))
    sns.barplot(x = 'Gender',y ='Payment',width = 0.2,data = dataframe,ax=ax,errorbar=None)
    ax.set_title('Payment Type Relate with Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Payment')
    ax.set_xticklabels(ax.get_xticklabels(),rotation =45)
    st.pyplot(fig)

   

    









