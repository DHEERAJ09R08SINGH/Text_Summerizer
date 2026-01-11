from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load Model + Tokenizer (from your trained folder OR t5-small)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def summarize_text(text):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    input_ids = inputs.input_ids.to(device)

    summary_ids = model.generate(
        input_ids,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        summary = summarize_text(input_text)
    return render_template("index.html", result=summary)


if __name__ == "__main__":
    app.run(debug=True)
