<?php
	for ($i=0; $i < 30; $i++) { 
		$treug[] = ($i + 1) * ($i + 2) / 2;
	}
	for ($i=0; $i < 30; $i++) { 
		$kvd[] = ($i + 1) * ($i + 1);
	}

	$i = 0;
	print "<table border = 1 cellpadding = 1>";
	while ($i < 30) {
		$j = 0;
		print "<tr>";
		while ($j < 30) {
			$res = ($j + 1) * ($i + 1);
			if (in_array($res, $treug) && in_array($res, $kvd)) {
				print "<td bgcolor = red><font size = 1>$res</font></td>";
			} elseif (in_array($res, $kvd)) {
				print "<td bgcolor = blue><font size = 1>$res</font></td>";
			} elseif (in_array($res, $treug)) {
				print "<td bgcolor = green><font size = 1>$res</font></td>";
			}
			else {
				print "<td bgcolor = white><font size = 1>$res</font></td>";
			}
			$j++;
		}
		print "</tr>";
		$i++;
	}
	print "</table>";
?>