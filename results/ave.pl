#! /usr/bin/perl
my $in = 'kdv-qt.dat';
open(IN,'<',$in) or die;
local $/ = "# ----------------\n";
my $gh = {};
while(<IN>) {
	my @lines = split("\n",$_);
	next if ($#lines == 0);
	my $h = parse(@lines);
	my $key = join('--',$h->{N},$h->{dt},$h->{T});
	for(my $i=0;$i<=$#{$h->{processor}};$i++) {
		my $p = $h->{processor}->[$i];
		my $t = (split(' ',$h->{elapsedTime}->[$i]))[0];
		push(@{$gh->{$key}->{$p}}, $t);
	}
}
close(IN);
for my $key (keys %{$gh}) {
	for my $processor (keys %{$gh->{$key}}) {
		my $a = $gh->{$key}->{$processor};
		my $ave = average($a);
		$gh->{$key}->{$processor} = $ave;
	}
	$gh->{$key}->{ratio} = $gh->{$key}->{CPU} / $gh->{$key}->{GPU};
	print $key, ' = ', $gh->{$key}->{ratio}, "\n";
}
exit 0;

sub average {
	my $a = shift;
	my $n = $#{$a} + 1;
	my $s = 0;
	for my $t (@{$a}) {
		$s += $t;
	}
	$s / $n;
}

sub parse {
	my @lines = @_;
	my $h = {};
	for my $line (@lines) {
		if ($line =~ /^#\s*(\w+)\s*=\s*(.*)/) {
			my ($key, $value) = ($1, $2);
			if (defined $h->{$key}) {
				if (ref($h->{$key}) ne 'ARRAY') {
					$h->{$key} = [$h->{$key}];
				}
				push(@{$h->{$key}}, $value);
			} else {
				$h->{$key} = $value;
			}
		}
	}
	$h;
}
