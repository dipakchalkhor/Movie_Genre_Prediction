from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("movie_genre_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    genre = None

    if request.method == "POST":
        title = request.form["title"]
        description = request.form["description"]

        text = title + " " + description

        # ML prediction
        genre = model.predict([text])[0]

    return render_template("index.html", genre=genre)


if __name__ == "__main__":
    app.run(debug=True)