
S=$1

rm -r fusion$S
for i in $(find fusion -iname '*.jpg'); do
	o=${i/fusion/fusion$S}
	o=${o/jpg/png}
	d=$(dirname $o)
	mkdir -p $d
	echo "SRAND=$RANDOM plambda $i \"x randn $S * +\" -o $o"
done | parallel

