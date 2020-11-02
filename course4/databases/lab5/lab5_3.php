<?php
	$breakfast = "gamburger";
	$breakfast2 = &$breakfast;
	print "$breakfast2<br/>";
	$breakfast = "tea";
	print "$breakfast2<br/>";
?>