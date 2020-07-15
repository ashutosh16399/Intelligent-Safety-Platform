import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#tokenizer = Tokenizer()


model=tf.keras.models.load_model("Toxicity.h5")
with open("tokenizer.pickle","rb") as file:
    tokenizer=pickle.load(file)
'''    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=120, padding='pre')
    predicted = model.predict(token_list)
    print(predicted)'''

seed_text = 'fuck off bitches '



g=tokenizer.texts_to_sequences([seed_text])
g=pad_sequences(g,maxlen=120,padding="post",truncating="post")
pred=model.predict(g)
print(pred)
f=0
Identity=0
                
f=1 if pred[0][0]>0.5 else 0
print(f)
print(seed_text)
if(f):
    print("----> its toxic")
else:
    pass