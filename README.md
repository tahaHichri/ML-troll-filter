# ML-troll-filter
A ML project for filtering spam and troll comments/emails.

## DESCRIPTION
Are you tired from people flooding your website/platform with spam and deceitful comments?
Well, you are not the only one.

If you were wondering what big platforms like Youtube, Tripavisor, and others have done to solve (or at list limit)<br/> this issue,
Well, two words: Machine Learning.
This project will help you train your machine to recognize potential spam messages by feeding it 16541 snippets of spam/ham messages.

Scikit-learn is required (check dependencies below) to train our <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier">Naive Bayes classifier</a>.

Dataset: We are using the first 3 parts of <a href="http://www2.aueb.gr/users/ion/data/enron-spam/">the Enron spam dataset</a> (minimize processing time).<br>

<i>If you would like to add even more accuracy to the model, you can add more parts to the data dir.</i>


## DEPENDENCIES
<ol>
<li>Python 3.7.+</li>
<li>numpy</li>

```
$ pip install numpy
```

<li>scikit-learn</li>

```
$ pip install scikit-learn
```

</ol>


## INSTALLATION
<ol>
<li>Clone this project</li>

```
$ git clone https://github.com/tahaHichri/ML-troll-filter.git
```

<li>Launch message keyword dictionary/features generations</li>

```
$ python classif.py 
```

<li>Check whether a message is a spam by passing it as an argument to <i>check.py</i>. The output should be "SPAM" or "NOT SPAM"</li>

```
$ python check.py "Hey, this is an example comment"
```

</ol>
 


