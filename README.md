# TextMiningIndeed
The train and test data is collected from a HackerRank competition by Indeed.
The code is created from scratch by me.
In the train data, job descriptions are given in the description column and corresponding tages such as bachelor's-degree-needed, 1-year-experience-needed, etc. has been added in the tags column.
The test data contains only the job descriptions and we have to assign the correct tags based on the job descriptions.
One job description can have multiple, single or no tags depending on the content.
The number of tags are limited and is 12 in number.
In the program, each tag is considered as one dependent variable and the descriptions are trained against the presence or absence of that tag (1/0).
Therefore, 12 sub training sets were generated for each tag. Each of the model makes prediction on the test data as whether that particular tag should be assigned to that description or not.
That's the overview. The description is first cleaned, preprocessed and stemmed. Then TFIDF vectorizer is used to vectorize the text of each description.
Thereafter, feature reduction was done using univariate analysis. Finally, the trimmed output was fed into the GBM to train and make predictions on test data.
Similar transformations were made for the test data.
