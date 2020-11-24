<?php
	$site = $_POST['site'];
	$list_sites = array("www.yandex.ru", "www.rambler.ru", "www.google.ru", "www.yahoo.ru", "www.altavista.ru");
	if (!in_array($site, $list_sites)) {
		print "<!DOCTYPE>
			<html>
			<head>
				<meta charset=utf-8>
				<title>lab8_5</title>
			</head>
			<body>
			<p>$site</p>
			<form action=lab8_5.php method=post>
				<select size=1 name=site>
				<option value=''>Поисковые системы:</option>";
				$i = 0;
				while ($i < count($list_sites)) {
					print "<option value=$list_sites[$i]>$list_sites[$i]</option>";
					$i++;
				}
				print "</select>
					<button type=submit>Перейти</button>
					</form>
					</body>
				</html>";
	} else {
		header("Location: http://$site");
	}

?>