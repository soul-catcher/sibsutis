<?php  
	$lang = $_GET['lang'];
	switch ($lang) {
		case 'ru':
			print "Russian";
			break;

		case 'en':
			print "English";
			break;

		case 'fr':
			print "France";
			break;

		case 'de':
			print "Germany";
			break;
		
		default:
			print "Язык не известен";
			break;
	}
?>