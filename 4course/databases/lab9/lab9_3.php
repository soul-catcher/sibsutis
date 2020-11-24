<?php
	$link = mysqli_connect("localhost", "root", "root", "sample");
	if (mysqli_connect_errno()) {
		printf("Не удалось подключиться: %s\n", mysqli_connect_error());
		exit();
	}
	$query = "SELECT * FROM notebook_br06";
	$result = mysqli_query($link, $query);
	if (mysqli_num_rows($result) > 0) {
		print "<html>
		   		<head>
		   			<meta charset=utf-8>
		   		</head>
		   		<body>
		   			<table border=1>
		   				<tr>
		   					<td><b>id</b></td>
		   					<td><b>name</b></td>
		   					<td><b>city</b></td>
		   					<td><b>address</b></td>
		   					<td><b>birthday</b></td>
		   					<td><b>mail</b></td>
		   				</tr>";
		while ($row = mysqli_fetch_assoc($result)) {
			print "<tr><td>" . $row['id'] . "</td><td>" . $row['name'] . "</td><td>" . $row['city'] . "</td><td>" . $row['address'] . "</td><td>" . $row['birthday'] . "</td><td>" . $row['mail'];
		}
		print "</td></tr></table></body></html>";
	} else {
		print "0 results";
	}
	mysqli_close($link);
?>