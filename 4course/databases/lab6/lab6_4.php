<?php
	$size = 7;

	$days = array("Воскресение", "Суббота", "Пятница", "Четверг", "Среда", "Вторник", "Понедельник");

	$colors = array("#ff0000", "#FF00FF",
					"#336699", "#0000ff",
					"#0000aa", "#DCDCDC",
					"#000000");

	function WeekDay($day, $color) {
		return "<font color = $color>$day</font><br/>";
	}

	for ($i=count($days); $i >= 0; $i--) {

		$str = WeekDay($days[$i], $colors[$i]);
		print "<font size = $size>$str</font><br/>";
		$size--;
	}
?>