# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
import os, PyPDF2, re
from typing import List

# Data handling
import pandas as pd

## Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor 
from sklearn.neighbors import KNeighborsRegressor

# Modelling Helpers
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# OpenAI
from openai import OpenAI

# Tokenization
import nltk
import tiktoken

# Langchain
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

# Transformers
from transformers import pipeline

# Matplotlib
import matplotlib.pyplot as plt

def return_r2(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    return r2, mse, mae, rmse

def regression_model_test(X_train, y_train, X_test, y_test):
    """
    """
    # Collect all R2 Scores.
    R2_Scores = []
    models = ['Linear Regression', 'Lasso Regression', 'AdaBoost Regression', 
            'Ridge Regression', 'GradientBoosting Regression',
            'RandomForest Regression', 'KNeighbours Regression']
    
    """ Linear Regression """
    clf_lr = LinearRegression()
    clf_lr.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_lr, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_lr.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Linear Regression model...")
    R2_Scores.append(r2)

    """ Lasso Regression """
    clf_la = Lasso()
    clf_la.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_la, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_la.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Lasso Regression model...")
    R2_Scores.append(r2)

    """ AdaBoostRegressor """
    clf_ar = AdaBoostRegressor(n_estimators=1000)
    clf_ar.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_ar, 
                                 X = X_train, y = y_train, 
                                 cv = 5, verbose = 0)
    y_pred = clf_ar.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed AdaBoost Regression model...")
    R2_Scores.append(r2)

    """ Ridge Regression """
    clf_rr = Ridge()
    clf_rr.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_rr, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_rr.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Ridge Regression model...")
    R2_Scores.append(r2)

    """ Gradient Boosting Regression """
    clf_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                        max_depth=1, random_state=0, 
                                        loss='squared_error', verbose=0)
    clf_gbr.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_gbr, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_gbr.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, y_pred)

    # Append to R2_Scores
    print("Completed Gradient Boosting Regression model...")
    R2_Scores.append(r2)

    """ Random Forest """
    clf_rf = RandomForestRegressor()
    clf_rf.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_rf, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_rf.predict(X_test)

    # Fine Tune Random Forest
    no_of_test=[100]
    params_dict={'n_estimators':no_of_test,'n_jobs':[-1],
                 'max_features':["auto",'sqrt','log2']}
    clf_rf=GridSearchCV(estimator=RandomForestRegressor(), 
                        param_grid=params_dict,scoring='r2')
    clf_rf.fit(X_train,y_train)

    pred=clf_rf.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, pred)

    # Append to R2_Scores
    print("Completed Random Forest Regression model...")
    R2_Scores.append(r2)

    """ KNeighbors Regression """
    clf_knn = KNeighborsRegressor()
    clf_knn.fit(X_train , y_train)
    accuracies = cross_val_score(estimator = clf_knn, 
                                 X = X_train, y = y_train, 
                                 cv = 5,verbose = 0)
    y_pred = clf_knn.predict(X_test)

    # Fine Tune KNeighbors
    n_neighbors=[]
    for i in range (0,50,5):
        if(i!=0):
            n_neighbors.append(i)
    params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}
    clf_knn=GridSearchCV(estimator=KNeighborsRegressor(), 
                         param_grid=params_dict,scoring='r2')
    clf_knn.fit(X_train,y_train)

    pred=clf_knn.predict(X_test)

    # Use function to return r2
    r2, mse, mae, rmse = return_r2(y_test, pred)

    # Append to R2_Scores
    print("Completed KNeighbors Regression model...")
    R2_Scores.append(r2)

    # Return Results
    print("---------------------")
    print("Finalizing results...")
    compare = pd.DataFrame({'Algorithms' : models , 'R2-Scores' : R2_Scores})
    
    return compare.sort_values(by='R2-Scores' ,ascending=False)

def get_diamond_info():
    """
    Returns cleaned_text as letters and numbers from a list of pdfs in 
    resources/pdfs folder
    """
    pdf_list = os.listdir('resources/pdfs')
    # A large block of text to vectorize as embeddings
    text = ""

    for pdf in pdf_list:
    # Open the PDF file in binary mode
        with open(f'resources/pdfs/{pdf}', 'rb') as file:
            # Create a PDF file reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Loop through each page in the PDF
            for page_num in range(len(pdf_reader.pages)):
                # Get the page object
                page = pdf_reader.pages[page_num]
                
                # Extract text from the page
                text += page.extract_text()

        text += "\n"

        text += """The most famous type of diamond cut is the round brilliant cut.
            It is known for its ability to maximize a diamond's sparkle and brilliance due to its precise faceting. 
            Diamond certification is a process in which a diamond is evaluated and graded based on its characteristics such as carat weight, cut, color, and clarity (known as the 4Cs). 
            This evaluation is conducted by a reputable gemological laboratory, such as the Gemological Institute of America (GIA) or the International Gemological Institute (IGI).
            A certificate detailing these characteristics is provided to verify the diamond's quality and authenticity. The diamond industry is vast, and there isn't one definitive "top diamond dealer" globally. 
            However, some of the largest and most well-known diamond companies include De Beers, Alrosa, and Rio Tinto. These companies are known for their extensive involvement in diamond mining and trade.
            Diamond impurities or inclusions are natural features within a diamond that can affect its clarity. Signs of diamond impurities include internal flaws such as feathers, crystals, or needles.
            Surface blemishes may include pits or scratches. These imperfections are often evaluated and rated on a clarity scale, ranging from "Flawless" to "Included."
            Blood diamonds, also known as conflict diamonds, are diamonds mined in war zones and sold to finance armed conflict against governments. 
            The term gained widespread awareness due to humanitarian concerns related to unethical mining practices and exploitation.
            Efforts such as the Kimberley Process have been established to prevent the trade of conflict diamonds.
            The diamond industry has a long and storied history. Diamonds were first discovered and mined in India around the 4th century BCE. 
            The industry expanded with the discovery of diamond deposits in South Africa in the late 1800s, which led to the establishment of large diamond mining companies like De Beers. 
            Today, diamonds are sourced from various regions worldwide, including Russia, Australia, and Canada.
            Major players in the diamond market include companies involved in diamond mining, trading, and retailing. De Beers, Alrosa, and Rio Tinto are some of the leading diamond mining companies. 
            In terms of retail, companies like Tiffany & Co., Cartier, and Graff Diamonds are well-known luxury brands in the diamond jewelry market.
            The number of diamonds used in high-end jewelry can vary depending on the design and size of the piece. 
            For example, a solitaire engagement ring may feature a single prominent diamond, while a necklace or bracelet might include multiple smaller diamonds set in intricate patterns. 
            High-end jewelry pieces often focus on quality and artistry, with diamonds chosen for their exceptional cut, clarity, color, and carat weight.
            Langsmith is likely a reference to an expert or resource specializing in diamond appraisals. 
            An expert like Langsmith can teach us how to evaluate diamonds based on the 4Cs (carat, cut, color, and clarity), how to identify diamond treatments or enhancements, and how to properly assess the value of a diamond. 
            They may also provide guidance on recognizing reputable grading certificates and authenticating diamonds.
            """
    # cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.]|(\s+)', lambda x: ' ' if x.group(1) else '', text)
    cleaned_text = cleaned_text.replace("\r","")
    cleaned_text = cleaned_text.replace("\n","")
    cleaned_text = cleaned_text.replace("\t","")
    cleaned_text = cleaned_text.strip()
    with open('resources/diamond_information.txt', 'w') as file:
        file.write(cleaned_text)
        file.close()
    return cleaned_text