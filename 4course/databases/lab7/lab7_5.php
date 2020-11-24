<?php
	$cust = array("cnum" => 2001, "cname" => "Hoffman", "city" => "London", "sname" => 1001, "rating" => 100);
	foreach ($cust as $key => $value) {
		print "$key => $value</br>";
	}
	print "</br></br>";
	asort($cust);
	print_r($cust);
	print "</br></br>";
	ksort($cust);
	print_r($cust);
	print "</br></br>";
	sort($cust);
	print_r($cust);
?>