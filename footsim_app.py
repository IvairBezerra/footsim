import streamlit as st
import sklearn
import pickle 

#título
st.title("Simule suas apostas com o FOOTSIM")

#cabeçalho
st.subheader("Faça seu jogo")

#apelido do jogo
bet_input = st.sidebar.text_input("De um apelido para seu jogo")

#escrevendo o nome do usuário
st.write("Jogo:", bet_input)

#dados dos usuários com a função

def get_user_date():
    pregnancies = st.sidebar.slider("Gravidez",0, 15, 1)
    glicose = st.sidebar.slider("Glicose", 0, 200, 110)
    blood_pressure = st.sidebar.slider("Pressão Sanguínea", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Espessura da pele", 0, 99, 20)
    insulin = st.sidebar.slider("Insulina", 0, 900, 30)
    bni= st.sidebar.slider("Índice de massa corporal", 0.0, 70.0, 15.0)
    dpf = st.sidebar.slider("Histórico familiar de diabetes", 0.0, 3.0, 0.0)
    age = st.sidebar.slider ("Idade", 15, 100, 21)
    
user_input_variables = get_user_date()

#grafico
graf = st.bar_chart(user_input_variables)