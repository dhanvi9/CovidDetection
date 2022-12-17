window.onload = initfunction;
function initfunction() {
	
	socket.onmessage = onMessage;
}
function onMessage(event) {
	console.log("my dialog ready" + $('#myDialog').is(':visible'));
	if($('#myDialog').is(':visible')==true)
		{
		$("#myDialog").dialog("close");
		}
	
	console.log("recieved some data from server...");
	console.log(event.data);
	var status=JSON.parse(event.data);
	console.log(status.event);
	switch(status.event)
	{
	case "patternsbydate":
		     console.log("message received for pattern");
		     patterns.series[0].setData(status.hourlyc);
		     patterns.series[1].setData(status.hourlyv);
	}
}

function fetchPatterns()
	{
	var date = document.getElementById("patterns_date").value;
	var message = '{"sender" : "connection by client", "event" : "dateForPattern","date":"'+ date+ '"}';//send JSON format message
	showProgress();
	console.log(message+"-----send for fetch patterns search");
	socket.send(message);//send the selected date to server to fetch data
	showProgress();
	}
function showProgress()
{
	console.log('show progress called');
	$("#myDialog").dialog({
		height : 100,
		dialogClass : 'no-close',
		width : 500,
		autoOpen : true,
		model : true,
		title : "Please wait.."
	});
	$('#myProgressBar')
			.progressbar({
				value : false
			});
	}
