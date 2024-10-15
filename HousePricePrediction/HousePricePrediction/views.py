
from django.shortcuts import render
import numpy as np
import  pandas as pd
import json
import  matplotlib.pyplot as plt
import seaborn as sns
import  sklearn.datasets
from sklearn.model_selection import  train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Reset session or state if needed
    request.session.flush()  # Clears the session

    Data = pd.read_csv(r"C:\Users\HP ProBook 440 G7\Desktop\MachineLearning model to PredictHousePRICE\nigeria_houses_data.csv")
    #print(Data)
    Data['title'].unique()
    Data = Data.drop(['parking_space'], axis=1)
    # convert title to numerical value using dictionary mapping
    title_mapping = {
        'Detached Duplex': 1,
        'Terraced Duplexes': 2,
        'Semi Detached Duplex': 3,
        'Detached Bungalow': 4,
        'Block of Flats': 5,
        'Semi Detached Bungalow': 6,
        'Terraced Bungalow': 7

    }
    Data['title'] = Data['title'].map(title_mapping)

    # CONVERYING TOWN TO NUMERICAL VALUE
    Data['town'].unique()

    location_mapping = {
        'Mabushi': 1,
        'Katampe': 2,
        'Lekki': 3,
        'Ajah': 4,
        'Epe': 5,
        'Lokogoma District': 6,
        'Oredo': 7,
        'Victoria Island (VI)': 8,
        'Mowe Ofada': 9,
        'Ikeja': 10,
        'Ikoyi': 11,
        'Magodo': 12,
        'Kaura': 13,
        'Galadimawa': 14,
        'Gwarinpa': 15,
        'Abeokuta North': 16,
        'Lugbe District': 17,
        'Ibeju Lekki': 18,
        'Yaba': 19,
        'Sango Ota': 20,
        'Ifako-Ijaiye': 21,
        'Agege': 22,
        'Ikorodu': 23,
        'Jahi': 24,
        'Ibadan': 25,
        'Orozo': 26,
        'Ifo': 27,
        'Owerri North': 28,
        'Guzape District': 29,
        'Idu Industrial': 30,
        'Owerri Municipal': 31,
        'Isheri North': 32,
        'Utako': 33,
        'Port Harcourt': 34,
        'Kuje': 35,
        'Isheri': 36,
        'Life Camp': 37,
        'Ipaja': 38,
        'Ado-Odo/Ota': 39,
        'Dape': 40,
        'Mushin': 41,
        'Ejigbo': 42,
        'Isolo': 43,
        'Ojodu': 44,
        'Gaduwa': 45,
        'Enugu': 46,
        'Dakwo': 47,
        'Asokoro District': 48,
        'Alimosho': 49,
        'Sagamu': 50,
        'Chikun': 51,
        'Egbeda': 52,
        'Wuye': 53,
        'Kubwa': 54,
        'Shomolu': 55,
        'Ogudu': 56,
        'Owerri West': 57,
        'Ibafo': 58,
        'Surulere': 59,
        'Obio-Akpor': 60,
        'Ayobo': 61,
        'Apo': 62,
        'Mowe Town': 63,
        'Ibadan South-West': 64,
        'Wuse 2': 65,
        'Durumi': 66,
        'Simawa': 67,
        'Arepo': 68,
        'Ikotun': 69,
        'Oluyole': 70,
        'Maitama District': 71,
        'Maryland': 72,
        'Ido': 73,
        'Karsana': 74,
        'Wuse': 75,
        'Ilorin West': 76,
        'Kurudu': 77,
        'Karmo': 78,
        'Abeokuta South': 79,
        'KM 46': 80,
        'Gbagada': 81,
        'Idimu': 82,
        'Kaduna South': 83,
        'Magboro': 84,
        'Gudu': 85,
        'Kukwaba': 86,
        'Mbora (Nbora)': 87,
        'Obafemi Owode': 88,
        'Jabi': 89,
        'Karu': 90,
        'Ojo': 91,
        'Garki': 92,
        'Aba': 93,
        'Ogijo': 94,
        'Asaba': 95,
        'Uyo': 96,
        'Kosofe': 97,
        'Oyigbo': 98,
        'Karshi': 99,
        'Osogbo': 100,
        'Kado': 101,
        'Ilupeju': 102,
        'Afijio': 103,
        'Ketu': 104,
        'Ado-Ekiti': 105,
        'Ibadan North': 106,
        'Egor': 107,
        'Nyanya': 108,
        'Ibarapa North': 109,
        'Kyami': 110,
        'Ojota': 111,
        'Dutse': 112,
        'Nasarawa': 113,
        'Calabar': 114,
        'Kaduna North': 115,
        'Eleme': 116,
        'Kafe': 117,
        'Oshodi': 118,
        'Dakibiyu': 119,
        'Amuwo Odofin': 120,
        'Ijede': 121,
        'Bwari': 122,
        'Dekina': 123,
        'Jos South': 124,
        'Agbara-Igbesa': 125,
        'Ijaiye': 126,
        'Kagini': 127,
        'Ohaji/Egbema': 128,
        'Mpape': 129,
        'Apapa': 130,
        'Lagos Island': 131,
        'Gwagwalada': 132,
        'Ijoko': 133,
        'Diplomatic Zones': 134,
        'Akure': 135,
        'Ethiope West': 136,
        'Badagry': 137,
        'Kano': 138,
        'Ede South': 139,
        'Ilorin South': 140,
        'Oke-Aro': 141,
        'Oke-Odo': 142,
        'Ikot Ekpene': 143,
        'Kusada': 144,
        'Akinyele': 145,
        'Kabusa': 146,
        'Dei-Dei': 147,
        'Eket': 148,
        'Egbe': 149,
        'Udu': 150,
        'Ibadan North-East': 151,
        'Danja': 152,
        'Yewa South': 153,
        'Warri': 154,
        'Duboyi': 155,
        'Jikwoyi': 156,
        'Oyo West': 157,
        'Agbara': 158,
        'Ovia North-East': 159,
        'Yenagoa': 160,
        'Central Business District': 161,
        'Guzamala': 162,
        'Lokoja': 163,
        'Jos North': 164,
        'Orile': 165,
        'Wumba': 166,
        'Ibadan North-West': 167,
        'Eko Atlantic City': 168,
        'Imota': 169,
        'Ikwerre': 170,
        'Keffi': 171,
        'Ilorin East': 172,
        'Uvwie': 173,
        'Umuahia': 174,
        'Ijesha': 175,
        'Okene': 176,
        'Aniocha South': 177,
        'Ijebu Ode': 178,
        'Okpe': 179,
        'Ughelli South': 180,
        'Ewekoro': 181,
        'Nassarawa': 182,
        'Mararaba': 183,
        'Uhunmwonde': 184,
        'Ughelli North': 185,
        'Ibeju': 186,
        'Abraka': 187,
        'Paikoro': 188,
        'Ikpoba Okha': 189,


    }
    Data['town'] = Data['town'].map(location_mapping)

    Data['state'].unique()

    state_mapping = {
        'Abuja': 1,
        'Lagos': 2,
        'Edo': 3,
        'Ogun': 4,
        'Oyo': 5,
        'Imo': 6,
        'Anambara': 7,
        'Rivers': 8,
        'Enugu': 9,
        'Kaduna': 10,
        'Kwara': 11,
        'Nasarawa': 12,
        'Abia': 13,
        'Delta': 14,
        'Akwa Ibom': 15,
        'Osun': 16,
        'Ekiti': 17,
        'Cross River': 18,
        'Kogi': 19,
        'Plateau': 20,
        'Kano': 21,
        'Katsina': 22,
        'Bayelsa': 23,
        'Borno': 24,
        'Niger': 25
    }
    Data['state'] = Data['state'].map(state_mapping)

    with open('title_mapping.json', 'r') as f:
        title_mapping = json.load(f)

    with open('location_mapping.json', 'r') as f:
        location_mapping = json.load(f)

    with open('state_mapping.json', 'r') as f:
        state_mapping = json.load(f)


    # check for missing value
    # print( Data.isnull().sum())
    # print("Number of columns and rows :", Data.shape)

    # labelling the data and target
    X = Data.drop(['price'], axis=1)
    Y = Data['price']
    # print(X)
    # print(Y)

    # split data into trianing n testing
    X_trian, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    # print(X.shape,X_trian.shape,X_test.shape)

    # the MODEL
    model = LinearRegression()
    # training the model with X_trian and Y_train
    model.fit(X_trian, Y_train)

    # Normalize user inputs to title case (e.g., 'lagos' -> 'Lagos')
    state_input = request.GET['n1'].title()  # Convert to title case
    town_input = request.GET['n2'].title()  # Convert to title case
    house_type_input = request.GET['n3'].title()  # Convert to title case

    # Get values from request and map to numeric values
    var1 = state_mapping.get(state_input)  # state
    var2 = location_mapping.get(town_input)  # town
    var3 = title_mapping.get(house_type_input)  # title (house type)
    var4 = float(request.GET['n4'])  # numeric inputs
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])

    # Ensure all mapped values are present and valid
    if var1 is None or var2 is None or var3 is None:
        return render(request, "predict.html", {"result2": "Invalid input data. Please try again."})

    pred = model.predict(np.array([var1,var2,var3,var4,var5,var6]).reshape(1,-1))

    pred = round(abs(pred[0]))

    # Return the predicted price
    price = "The predicted price is \u20A6 "+str(pred)

    return render(request,"predict.html", {"result2": price})
