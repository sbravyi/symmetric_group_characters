# measure the runtime for computing a given column of the character table of the symmetric group S_n

LoadPackage("ctbllib");



# permutation that defines a column of the character table
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18)(19,20);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18)(19,20)(21,22);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18)(19,20)(21,22)(23,24);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18)(19,20)(21,22)(23,24)(25,26);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18)(19,20)(21,22)(23,24)(25,26);
#sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18)(19,20)(21,22)(23,24)(25,26)(27,28);
sigma:=(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)(13,14)(15,16)(17,18)(19,20)(21,22)(23,24)(25,26)(27,28)(29,30);
n:=Size(ListPerm(sigma));
Print("n=");
Print(n);
Print("\n");
Print("group element=");
Print(sigma);
Print("\n");

G:=SymmetricGroup(n);
irrG:=Irr(G);
Print("Begin computation");
Print("\n");

startTime := Runtime();
for irrep in irrG do
chi:=(sigma^irrep);
#Print("chi=");
#Print(chi);
#Print("\n");
od;

Print("Done");
Print("\n");
endTime := Runtime();
runtime:= endTime - startTime;
Print("runtime=");
Print(runtime);
Print("\n");
