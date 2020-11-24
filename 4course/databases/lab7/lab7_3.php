<?php
	$i = 0;
	print "<table border = 1 cellpadding = 0>";
	while ($i < 30) {
		$j = 0;
		print "<tr>";
		while ($j < 30) {
			$res = ($j + 1) * ($i + 1);
			switch ($res % 7) {
				case 0:
					print "<td  width = 14 height = 15 bgcolor = white><font size = 1>&nbsp</font></td>";
					break;
				
				case 1:
					print "<td  width = 14 height = 15 bgcolor = aqua><font size = 1>&nbsp</font></td>";
					break;

				case 2:
					print "<td  width = 14 height = 15 bgcolor = blue><font size = 1>&nbsp</font></td>";
					break;

				case 3:
					print "<td  width = 14 height = 15 bgcolor = yellow><font size = 1>&nbsp</font></td>";
					break;

				case 4:
					print "<td  width = 14 height = 15 bgcolor = purple><font size = 1>&nbsp</font></td>";
					break;

				case 5:
					print "<td  width = 14 height = 15 bgcolor = red><font size = 1>&nbsp</font></td>";
					break;

				case 6:
					print "<td  width = 14 height = 15 bgcolor = lime><font size = 1>&nbsp</font></td>";
					break;
			}
			$j++;
		}
		print "</tr>";
		$i++;
	}
	print "</table>";
?>