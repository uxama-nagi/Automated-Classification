<h1 style="text-align:center">Automated Classification of Performance Issues in Open-Source Repositories</h1><br>



Step 1:<br>
Download python in you system and make sure it is added to a system's path in <b>"System Environments Variables"</b>

Step 2:<br>
Download the requirements.txt file. After downloading open the <b>"cmd"</b> and go to the path where you have downloaded this file

Step 3:<br>
Run this command to download all required libraries: <br>
<code>pip install -r requirements.txt</code>

Step 4:<br>
If you want to run the mining-tool.py to search and mine repositories type this code in cmd 
<code>python mining-tool.py "repo:myorg/myrepo" "your-access-token" bug output 10000</code></br>
Replace <code>"query" "your-access-token" [keywords] "output_dir" "number_rows_per_file</code> with your desired arguments

Link to dataset<br>
https://drive.google.com/drive/folders/1aXvGmVN9BM_1DCeN1-COZ48wsFOMbqCZ?usp=sharing <br>
<b>Note:</b> Make sure after downloading the data change the folder name. e.g. "Your Folder Name"<br>
change <code>data = "dataset.csv"</code> to <code>data = "Your Folder Name" </code> in model_creater.py file 
