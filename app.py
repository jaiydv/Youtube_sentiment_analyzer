from flask import Flask, render_template, request
import yt_comment_extractor as yt
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')

@app.route('/yt_analyzer',methods=["GET"])
def yt_analyzer():
    return render_template("yt_analyzer.html")

@app.route('/yt_form',methods=["GET","POST"])
def yt_form():
    if request.method == 'POST':
        print(request.form)
        platform = request.form.get('platform')
        method = request.form.get('method')
        url = request.form.get('url')
        total_comments = int(request.form.get('total_comments'))

        yt.yt_run(url,total_comments)

        positive_words = ['admiration', 'amusement', 'desire', 'approval', 'caring', 'curiosity', 'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'realization', 'relief', 'surprise']
        negative_words = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness']
        neutral_words = ['confusion', 'neutral']

        data = pd.read_csv("data.csv")
        label = data.iloc[:, 0].tolist()  # Extract values of the first column
        counts = data.iloc[:, 1].tolist() 
        
        emotion = ["positive","negative","neutral"]

        emotion_count = [0,0,0]
        for i in range(len(label)):
            if label[i] in positive_words:
                emotion_count[0] +=counts[i]
            elif i in negative_words:
                emotion_count[1]+=counts[i]
            else:
                emotion_count[2]+=counts[i]
        


    # print(request.form)
    return render_template("result.html",labels=label,values = counts,emotion=emotion,emotion_count=emotion_count)

if __name__ == "__main__":
    app.run(debug=True)