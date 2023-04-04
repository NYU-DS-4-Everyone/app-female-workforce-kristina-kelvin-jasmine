import streamlit as st
import pandas as pd
from PIL import Image
import requests
import json
import streamlit as st
from streamlit.components.v1 import html
import seaborn as sns
import pandas as pd
import numpy as np
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()


### The first two lines of the code load an image and display it using the st.image function.
image = Image.open('images/labor.jpeg')
st.image(image, width=800)

### The st.title function sets the title of the web application to "Mid Term Template - 01 Introduction Page 🧪".
st.title("Female Labor Force Participation Rate 👩🏼‍💻")


st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
#app_mode = st.sidebar.selectbox('🔎 Select Page',['Home'])

select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Labor Participation"])
df = pd.read_csv("Dataset_LaborPart3.csv")

app_mode = st.sidebar.selectbox('Select Page',['Summary 🚀','Visualization 📊','Prediction 📈'])

if app_mode == 'Summary 🚀':
    st.subheader("01 Summary Page - Spotify Data Analysis 🚀")

    st.info("Women make up **51% of the global population**. Yet, they have been chronically **underrepresented in the labor force**. This has led to the **lack of female autonomy** both in the public and the private life. ")
    st.info("Given this context, it is imperative that the causes behind the varying levels of female labor force participation rate be studied and understood. Only so the **right policy decisions** can be guided toward solving this issue.")

    st.info("On this interactive website you are invited to explore the factors influencing the rates of female labor force participation across the globe. Reflect on whether female participation in the labor market is the **driving force of economic development** or in contrast, a higher rate of female participation in the labor force is the **result of a developed economy**.")

    st.markdown("### 00 - Description of the Dataset")

    st.markdown("##### Explore the Dataset:")
    head = st.radio('View the **top** (head) or the **bottom** (tail) of the dataset', ('Head', 'Tail'))
    num = st.number_input('Select the number of rows to view:', 5, 158)
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    st.caption('Data Source - OECD: Gender, Institutions and Development Database 2009 (GID-DB)')

    st.markdown("##### Size of the Dataset:")
    st.text('(Number of Countries, Number of Variables)')
    st.write(df.shape)


    st.markdown("##### Description of each variable:")
    st.markdown(" **Country**: Country")
    st.markdown(" **Population**: Population")
    st.markdown(" **PopulationGrowth**: Population Growth (in %)")
    st.markdown(" **GDP_PerCapita**: GDP per capita (constant 2000 US$)")
    st.markdown(" **PrimaryEnrollment_ratio**: Ratio of female to male primary enrollment")
    st.markdown(" **LifeExpectancy_fem**: Women's life expectancy (in years)")
    st.markdown(" **Fertility**: Fertility rate")
    st.markdown(" **Parliament_fem**: Women in Parliament (as % of total)")
    st.markdown(" **EarnedIncome_fem**: Estimated earned income (PPP US$), female")
    st.markdown(" **EarnedIncome**: Estimated earned income (PPP US$), male")
    st.markdown(" **HumanDevelopmentIndex**: Human Development Index (Value), https://hdr.undp.org/data-center/human-development-index#/indicies/HDI")
    st.markdown(" **GenderEmpowermentMeasure**: Gender Empowerment Measure (Value), https://www.cbs.nl/en-gb/news/2009/27/dutch-women-among-the-most-emancipated-in-europe/gender-empowerment-measure--gem--")
    st.markdown(" **GenderDevelopmentIndex**: Gender-related Development Index (Value), https://hdr.undp.org/gender-development-index#/indicies/GDI")
    st.markdown(" **LaborParticipation**: Labor force participation rate, female (% of female population ages 15+) (national estimate)")


    st.markdown("##### Dataset - Countries by income per capita and population size")

    edges_gdp = [0, 1045, 4095, 12695, max(df['GDP_PerCapita'])]
    categories_gdp = ['Low Income', 'Lower Middle Income', 'Upper Middle Income', 'High Income']
    df['GDP_Group'] = pd.cut(df['GDP_PerCapita'], bins = edges_gdp, labels = categories_gdp)
    group_counts_gdp = df['GDP_Group'].value_counts()

    plt.pie(group_counts_gdp, labels=group_counts_gdp.index, autopct='%1.1f%%', startangle=90)
    circle = plt.Circle((0,0), 0.7, color='white')
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    plt.axis('equal')
    plt.title('Countries by GDP per capita')
    st.pyplot()
    st.info('The division of countries in the dataset according to GDP per capita.')
    st.write('')

    st.markdown("##### Female life expectancy by GDP per capita")

    df_sorted = df.sort_values(by=['LifeExpectancy_fem'], ascending=False)
    top_gdp_countries = df.sort_values(by='GDP_PerCapita', ascending=False).head(5)['Country'].tolist()
    bottom_gdp_countries = df.sort_values(by='GDP_PerCapita', ascending=True).head(5)['Country'].tolist()

    selected_countries = top_gdp_countries + bottom_gdp_countries
    selected_df = df.loc[df['Country'].isin(selected_countries)]

    stacked_df = selected_df.pivot(index='Country', columns='GDP_Group', values='LifeExpectancy_fem')
    stacked_df.plot(kind='bar', stacked=True)
    plt.title('Female Life Expectancy by GDP per capita')
    plt.xlabel('Countries')
    plt.ylabel('Female Life Expectancy')
    plt.legend(title='GDP Group')
    st.pyplot()

    st.info('The relationship between GDP per capita and female life expectancy.')
    st.write('')


    st.markdown("### 01 - Descriptive Statistics")
    st.dataframe(df.describe())


    st.markdown("### 02 - Missing Values")
    dfnull = df.isnull().sum()/len(df)*100
    avgmiss = dfnull.sum().round(2)
    st.write("Total percentage of missing values:",avgmiss)
    st.write(dfnull)
    if avgmiss <= 30:
        st.success("The missing values account for less then 30 percent of the dataset. Observe which variables have the highest number of missing observation. To which countries do the missing observations correspond to? What are the challenges related to the collection of statistical data? 🤔")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing values.")

    st.markdown("### 04 - Normality")
    image = Image.open('images/skewness.png')
    st.image(image, width = 800)
    aa= pd.DataFrame(df).skew()
    normalityskew= round(aa.mean(),4)
    if normalityskew == 0 :
        st.success("The dataset is **normally distributed**: Mean = Mode = Median")
    elif normalityskew > 0:
        st.success("The dataset is **positively skewed**:  Mean  >  Median  >  Mode")
    elif normalityskew < 0:
        st.success("The dataset is **negatively skewed**: Mode  >  Median  >  Mean")



if app_mode == 'Visualization 📊':
    st.subheader("Female Labor Force Participation Rate - Visualization 📊")

    df = pd.read_csv('Dataset_LaborPart3.csv')
    list_variables = df.columns

    ## Countries categories donut chart
    st.header('Dataset - Countries by income per capita and population size')

    edges_gdp = [0, 1045, 4095, 12695, max(df['GDP_PerCapita'])]
    categories_gdp = ['Low Income', 'Lower Middle Income', 'Upper Middle Income', 'High Income']
    df['GDP_Group'] = pd.cut(df['GDP_PerCapita'], bins = edges_gdp, labels = categories_gdp)
    group_counts_gdp = df['GDP_Group'].value_counts()


    plt.pie(group_counts_gdp, labels=group_counts_gdp.index, autopct='%1.1f%%', startangle=90)
    circle = plt.Circle((0,0), 0.7, color='white')
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    plt.axis('equal')
    plt.title('Countries by GDP per capita')

    edges_pop = [0, 10_000_000, 100_000_000, 500_000_000, 1_000_000_000, max(df['Population'])]
    categories_pop = ['<10M', '10M - 100M', '100M-500M', '500M - 1B', '1B<']
    df['Population_Group'] = pd.cut(df['Population'], bins=edges_pop, labels=categories_pop)
    group_counts_pop = df['Population_Group'].value_counts()


    plt.pie(group_counts_pop, labels = group_counts_pop.index, autopct='%1.1f%%', startangle=90)
    circle = plt.Circle((0,0), 0.7, color='white')
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    plt.axis('equal')
    plt.title('Countries by population size')

    st.pyplot()
    st.info('The division of countries in the dataset according to GDP per capita and population size.')
    st.write('')



    ## Female life expectancy bar chart
    st.header('Female life expectancy by GDP per capita')

    df_sorted = df.sort_values(by=['LifeExpectancy_fem'], ascending=False)
    top_gdp_countries = df.sort_values(by='GDP_PerCapita', ascending=False).head(5)['Country'].tolist()
    bottom_gdp_countries = df.sort_values(by='GDP_PerCapita', ascending=True).head(5)['Country'].tolist()

    selected_countries = top_gdp_countries + bottom_gdp_countries
    selected_df = df.loc[df['Country'].isin(selected_countries)]

    stacked_df = selected_df.pivot(index='Country', columns='GDP_Group', values='LifeExpectancy_fem')
    stacked_df.plot(kind='bar', stacked=True)
    plt.title('Female Life Expectancy by GDP per capita')
    plt.xlabel('Countries')
    plt.ylabel('Female Life Expectancy')
    plt.legend(title='GDP Group')
    st.pyplot()

    st.info('The relationship between GDP per capita and female life expectancy.')
    st.write('')

if app_mode == 'Prediction 📈':

    st.title("Female Labor Force Participation Rate - Linear Regression")
    df = pd.read_csv("Dataset_LaborPart3.csv")


    list_variables = ['Population', 'Population_Growth', 'GDP_PerCapita',
        'PrimaryEnrollment_ratio', 'LifeExpectancy_fem', 'Fertility',
        'Parliament_fem', 'EarnedIncome_fem', 'EarnedIncome',
        'HumanDevelopmentIndex', 'GenderEmpowermentMeasure',
        'GenderDevelopmentIndex', 'LaborParticipation']
    select_variable = st.sidebar.selectbox('Select Variable to Predict', list_variables)

    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    new_df = df[list_variables].drop(labels = select_variable, axis=1)
    list_var = new_df.columns

    output_multi = st.multiselect("Select Explanatory Variables",list_var ,["LaborParticipation","EarnedIncome"])

    new_df2 = new_df[output_multi]
    x = new_df2
    y = df[select_variable]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= train_size)


    lm = LinearRegression()
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)

    col1, col2 = st.columns(2)
    col1.subheader("Feature Columns top 25")
    col1.write(x.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))

    st.subheader('Results')

    st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
    st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
    st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))



st.markdown(" ")
st.markdown("##### 💻 **App Contributors: Kristina Sisiakova, Kelvin Delarosa, Jasmine Hus** ")

st.markdown(f"#####  Link to Project Website [here]({'https://github.com/NYU-DS-4-Everyone'}) 🚀 ")
st.markdown(f"#####  Feel free to contribute to the app and give a ⭐️")