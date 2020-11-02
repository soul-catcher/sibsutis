<?php
	define('NUM_E', 2.71828);
	printf("Число e равно %.5f<br/>", NUM_E);
	$num_e1 = NUM_E;
	print gettype($num_e1);
	echo "<br/>";
	settype($num_e1, "string");
	printf("%s %s</br>", gettype($num_e1), $num_e1);
	settype($num_e1, "integer");
	printf("%s %s</br>", gettype($num_e1), $num_e1);
	settype($num_e1, "bool");
	printf("%s %s</br>", gettype($num_e1), $num_e1);
?>