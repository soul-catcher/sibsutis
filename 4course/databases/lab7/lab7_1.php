<?php

	function print_array($array) {
		foreach ($array as $item) {
			print "$item&nbsp&nbsp";
		}
	}

	for ($i=0; $i < 10; $i++) { 
		$treug[] = ($i + 1) * ($i + 2) / 2;
	}
	print_array($treug);
	print "</br></br>";
	for ($i=0; $i < 10; $i++) { 
		$kvd[] = ($i + 1) * ($i + 1);
	}
	print_array($kvd);
	print "</br></br>";
	$res = array_merge($treug, $kvd);
	print_array($res);
	print "</br></br>";
	sort($res);
	print_array($res);
	print "</br></br>";
	unset($res[0]);
	print_array($res);
	print "</br></br>";

	$res1 = array_unique($res);

	print_array($res1);
	print "</br></br>";
?>