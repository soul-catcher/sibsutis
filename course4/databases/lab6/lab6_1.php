<?php 
	$i = 0;
	$color = "blue";
	print "<table border = 1 cellpadding = 5>";
	while ($i < 10) {
		$j = 0;
		print "<tr>";
		while ($j < 10) {
			$res = ($j + 1) * ($i + 1);
			if ($i == $j) {
				print "<td bgcolor = $color>$res</td>";
			} else {
				print "<td>$res</td>";
			}
			$j++;
		}
		print "</tr>";
		$i++;
	}
	print "</table>";
