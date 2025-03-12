import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from analysis import eda

logistic = joblib.load('logistic.pkl')

st.set_page_config(
    page_title="Grocery Sales Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

Branch_uni = np.array(['A', 'C' ,'B'])
Branch_uni = np.array(Branch_uni).reshape(-1,1)

City_uni = np.array(['Yangon' ,'Naypyitaw' ,'Mandalay'])
City_uni = np.array(City_uni).reshape(-1,1)

Customer_type_uni = np.array(['Member' ,'Normal'])
Customer_type_uni = np.array(Customer_type_uni).reshape(-1,1)

Gender_uni = np.array(['Female', 'Male'])
Gender_uni = np.array(Gender_uni).reshape(-1,1)

Product_line_uni = np.array(['Health and beauty', 'Electronic accessories' ,'Home and lifestyle',
 'Sports and travel' ,'Food and beverages' ,'Fashion accessories'])
Product_line_uni = np.array(Product_line_uni).reshape(-1,1)

Payment_uni = np.array(['Ewallet', 'Cash' ,'Credit card'])
Payment_uni = np.array(Payment_uni).reshape(-1,1)

branch_encode = LabelEncoder()
city_encode = LabelEncoder()
Customer_type_encode = LabelEncoder()
Gender_encode = LabelEncoder()
Product_line_encode = LabelEncoder()
Payment_encode= LabelEncoder()

branch_encode.fit(Branch_uni)
city_encode.fit(City_uni)
Customer_type_encode.fit(Customer_type_uni)
Gender_encode.fit(Gender_uni)
Product_line_encode.fit(Product_line_uni)
Payment_encode.fit(Payment_uni)


def predict_value(features):
    features = np.array(features).reshape(1,-1)
    pred = logistic.predict(features)
    return pred

st.sidebar.title('Grocery Sales Prediction')

st.sidebar.header('Overview')
st.sidebar.write('Welcome to the Grocery Sales Prediction tool! This application helps forecast grocery sales based on various factors influencing customer purchases.')

st.sidebar.header('Select Parameters')
st.sidebar.write('1. **Branch**: Select the grocery store branch to analyze sales trends for a specific location.')  
st.sidebar.write('2. **City**: Choose the city where the grocery store is located to account for regional variations.')  
st.sidebar.write('3. **Customer Type**: Select whether the customer is a "Member" or a "Normal" customer, as purchasing behavior differs.')  
st.sidebar.write('4. **Gender**: Choose the customer’s gender to analyze potential buying trends based on demographics.')  
st.sidebar.write('5. **Product Line**: Select the category of grocery items (e.g., Food & Beverages, Fashion Accessories, Electronics, etc.).')  

st.sidebar.header('Sales & Revenue Parameters')
st.sidebar.write('1. **Unit Price**: Enter the price per unit of the selected product to determine its impact on sales.')  
st.sidebar.write('2. **Tax (5%)**: Input the tax amount applied to the purchase.')  
st.sidebar.write('3. **Total Sale Value**: View the total purchase amount, including tax and product quantity.')  
st.sidebar.write('4. **Time of Purchase**: Select the time of purchase to observe sales trends at different hours of the day.')  
st.sidebar.write('5. **Payment Method**: Choose the payment mode (Cash, Credit Card, or E-Wallet) used by the customer.')  

st.sidebar.header('Financial & Performance Metrics')
st.sidebar.write('1. **Cost of Goods Sold (COGS)**: Input the cost incurred to sell the selected products.')  
st.sidebar.write('2. **Gross Margin**: The gross margin percentage to evaluate profitability.')  
st.sidebar.write('3. **Gross Income**: Revenue generated after deducting the cost of goods sold.')  
st.sidebar.write('4. **Customer Rating**: Analyze how customer satisfaction influences sales and future revenue projections.')  

st.sidebar.header('Additional Information')
st.sidebar.write('Use these parameters to generate grocery sales forecasts based on customer preferences, product pricing, and financial factors.')  
st.sidebar.write('Click the "Predict Sales" button to get insights into future grocery store sales.')  


option = st.sidebar.radio("Select a section",["Model Prediction","Data Analysis"])


if option == "Model Prediction":

    st.title("Grocery Sales Prediction")

    col1,col2,col3 = st.columns(3)

    with col1:

        Invoice_ID   = st.number_input('Invoice_ID',min_value = 101176199, max_value = 898042717)
        Branch         = st.selectbox('Branch',['A', 'C' ,'B'])  
        City        = st.selectbox('city',['Yangon' ,'Naypyitaw' ,'Mandalay'])      
        Customer_type    = st.selectbox('Customer_type',['Member' ,'Normal']) 
        Gender          = st.selectbox('Gender',['Female', 'Male'])  
        Product_line     = st.selectbox('Product_line',['Health and beauty', 'Electronic accessories' ,'Home and lifestyle',
        'Sports and travel' ,'Food and beverages' ,'Fashion accessories'])

    with col2:
        Unit_price    = st.number_input('Unit_price',min_value = 10.08, max_value = 99.96)
        Tax_5        = st.number_input('Tax_5%',min_value = 0.5085, max_value = 49.65)
        Total          = st.number_input('Total',min_value = 10.6785, max_value = 1042.65)
        Time            = st.number_input('Time',min_value = 1000, max_value = 2059)
        Payment        = st.selectbox('Payment',['Ewallet', 'Cash' ,'Credit card'])   
        cogs           = st.number_input('cogs',min_value = 10.17, max_value = 993.0)

    with col3:

        gross_margin   = st.number_input('gross_margin',min_value = 4.761904762, max_value = 4.761904762)
        gross_income   = st.number_input('gross_income',min_value = 0.5085, max_value = 49.65)
        Rating         = st.number_input('Rating',min_value = 4.0, max_value = 10.0)
        day              = st.number_input('day',min_value = 1, max_value = 31)
        month            = st.number_input('month',min_value = 1, max_value = 3)
        year   = st.number_input('year',min_value = 2019, max_value = 2019)

    encode_branch = branch_encode.transform([[Branch]])[0]
    encode_city = city_encode.transform([[City]])[0]
    encode_customer=Customer_type_encode.transform([[Customer_type]])[0]
    encode_gender =Gender_encode.transform([[Gender]])[0]
    encode_product =Product_line_encode.transform([[Product_line]])[0]
    encode_payment =Payment_encode.transform([[Payment]])[0]

    features = [
        Invoice_ID,   
        encode_branch,         
        encode_city,            
        encode_customer,   
        encode_gender,           
        encode_product,    
        Unit_price,    
        Tax_5,       
        Total,          
        Time,            
        encode_payment,         
        cogs,          
        gross_margin,   
        gross_income,   
        Rating,        
        day,              
        month,                       
        year 
    ]

    features = np.array(features).reshape(1, -1)

    if st.button("Predict"):
        result = predict_value(features)
        st.write("Predict Quantity",result)

elif option == "Data Analysis":
    st.title('Data Analysis(EDA)')
    dataframe = pd.read_csv('data.csv')
    eda(dataframe)
