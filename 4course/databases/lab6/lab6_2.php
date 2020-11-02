<?php 
	$color = "blue";
	print "<table border = 1 cellpadding = 5>";
	for ($i=0; $i <= 10; $i++) { 
		print "<tr>";
		for ($j=0; $j <= 10; $j++) { 
			$res = $i + $j;
			if ($res == 0) {
				print "<td><font color = red>+</font></td>";
			} elseif ($i == 0 || $j == 0) {
				print "<td><font color = $color>$res</font></td>";
			} else {
				print "<td>$res</td>";
			}
		}
		print "</tr>";
	}
	print "</table>";
?>