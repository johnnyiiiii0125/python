#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:26:48 2023

@author: Merlin
"""

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import nltk
import jieba
import jieba.posseg
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def tag_text():
    text = input_text.get("1.0", tk.END)
    language = language_var.get()

    if language == "English":
        words = word_tokenize(text)
        tagged_words = nltk.pos_tag(words)
    elif language == "中文":
        words = jieba.lcut(text)
        tagged_words = jieba.posseg.lcut(text)

    output_text.delete("1.0", tk.END)
    for word, tag in tagged_words:
        output_text.insert(tk.END, f"{word}_{tag} ")

def copy_to_clipboard():
    result = output_text.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(result)

root = tk.Tk()
root.title("Part-of-Speech Tagger v2.0")

# Add label for input text box
input_label = tk.Label(root, text="输入英文或中文，限2000词")

input_label.pack(padx=10, pady=10)

input_text = ScrolledText(root, wrap=tk.WORD, width=50, height=10)
input_text.pack(padx=10, pady=10)

language_var = tk.StringVar()
language_var.set("English")
language_menu = ttk.OptionMenu(root, language_var, "English", "English", "中文")
language_menu.pack(padx=10, pady=10)

tag_button = ttk.Button(root, text="Tag it!", command=tag_text)
tag_button.pack(pady=5)

output_text = ScrolledText(root, wrap=tk.WORD, width=50, height=10)
output_text.pack(padx=10, pady=10)

copy_button = ttk.Button(root, text="复制", command=copy_to_clipboard)
copy_button.pack(pady=5)

root.mainloop()
