<?php  
	$lang = $_GET['lang'];
	if ($lang == "ru") {
		print "Russian";
	} elseif ($lang == "en") {
		print "English";
	} elseif ($lang == "fr") {
		print "France";
	} elseif ($lang == "de") {
		print "Germany";
	} else {
		print "Язык не известен";
	}
?>