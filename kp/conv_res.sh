n=126
echo $#
if [ $# >0 ]; then
	echo $1
	n=$1;
fi

cd res/
cp ../conv.py .

i=0
while [[ $i != $n ]]; do
	echo "conv $i.data"
	./morozov_conv.py $i.data $i.jpg
	i=$(($i+1))
done

rm conv.py
cd ..
	
