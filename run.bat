
set /A client_counts=10

start cmd.exe /k c:\users\Afrooz\anaconda3\envs\Capstone\python.exe "D:\UW\Final thesis\audio_classification\server.py" %client_counts%

TIMEOUT /T 10 

set /A one = 1
set /A count= %client_counts% - %one%

FOR /L %%K IN (0,1,%count%) DO (
	start cmd.exe /c c:\users\Afrooz\anaconda3\envs\Capstone\python.exe "D:\UW\Final thesis\audio_classification\client.py" %%K %client_counts%
)

pause