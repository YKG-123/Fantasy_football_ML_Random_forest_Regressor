This GitHub houses my code for both the website ie. HTML and CSS code and the source code for how I trained and validated my models.
There are two folders one titled Notebooks and one titled htmls the one titled htmls houses all the code for the github pages site, and the one titled notebooks holds all of my python notebooks for this project.
The notebooks titled Xposition_postable are the notebooks holding the code to generate the HTML code for the website.
The notebooks that house my source code for the model are titled X position_Prev_Season_1.ipynb for example RB_Prev_Season_1.ipynb. The rest of this README will be specifically for the running backs but similar proccesses were taken for all of them.
I start by inputting NFL data from the last 10 years from Kaggle and a site that holds ADP (average draft position) data from previous years.
I then clean the data so they all have similar names so I can merge the data set into one which I then do.
I then filter out the data that I want I limited it to greater than 5 ppg since there were so many players that rarely played effecting the mmodel and I only wanted it to predict useful players for fantasy football
I then made sure all of my values were the correct data types and aggregated correctly then I offset the data and add in the features to the model.
I trained and ran the model using a 70-30 train test split.
I then made a model for rookies using a similar method but had less features to work with obviously because they did not have the previous year's stats.
I then input the data for this year.
I ran into the problem that the rookie model and the normal model had different means and standard deviations so i centered both around the mean and standard dev of the running back position over the past 10 years for players scoring 5> points per game.
I then made some scatter plots to analyze how the data unfolds and to see where one can get some value in the draft.
I then uploaded the prediction data as a csv to my google drive and inputed it into my postable docs.
