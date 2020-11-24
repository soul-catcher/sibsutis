<?php
	$valign = $_POST['valign'];
	$align = $_POST['align'];
	if (is_null($valign)) {
		$valign = "middle";
	}
	if (is_null($align)) {
		$align = "center";
	}
	print "<div align=center> <table border=1><td align=$align valign=$valign height=100 width=100>Text</td></table></br>";
	print "<a href=lab8_1.html>Назад</a></div>"
?>