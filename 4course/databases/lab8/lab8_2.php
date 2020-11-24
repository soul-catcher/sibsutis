<?php
	if (!$_POST) {
		$align = "left";
		$valign = "top";
	} else {
		$valign = $_POST['valign'];
		$align = $_POST['align'];
		if (!isset($valign)) {
			$valign = "top";
		}
		if (!isset($align)) {
			$align = "left";
		}
	}
	print "<!DOCTYPE html>
	<html>
	<head>
		<meta charset=utf-8>
		<title>Form</title>
	</head>
	<body>
		<div align=center> <table border=1><td align=$align valign=$valign height=100 width=100>Text</td></table>
		<form action='' method=post>
			<p><b>Выберите горизонтальное расположение:</b></br></p>
			<input type=radio name=align
			id=horizontalChoise value=left checked=true>
			<label for=horizontalChoise>слева</br></label>

			<input type=radio name=align
			id=horizontalChoise1 value=center>
			<label for=horizontalChoise1>по центру</br></label>

			<input type=radio name=align
			id=horizontalChoise2 value=right>
			<label for=horizontalChoise2>справа</br></label>

			<p><b>Выберите вертикальное расположение:</b></br></p>
			<input type=checkbox name=valign
			id=verticalChoise value=top>
			<label for=verticalChoise>сверху</br></label>

			<input type=checkbox name=valign
			id=verticalChoise1 value=middle>
			<label for=verticalChoise1>посередине</br></label>

			<input type=checkbox name=valign
			id=verticalChoise2 value=bottom>
			<label for=verticalChoise2>внизу</br></br></label>

			<button type=submit>Выполнить</button>
		</form>
	</body>
	</html>";
?>