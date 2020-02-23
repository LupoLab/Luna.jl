import Test: @test, @testset, @test_throws
import Luna: Utils

@testset "Utils" begin

@testset "Chunking" begin
    function contains_all_unique(chunks, x)
        contains_unique = []
        for xi in x
            c = count([xi in chi for chi in chunks])
            push!(contains_unique, (c==1))
        end
        return all(contains_unique)
    end
    pass = []
    for L = 5:11:896
        for n = 1:24
            x = collect(1:L)
            ch = Utils.chunks(x, n)
            push!(pass, contains_all_unique(ch, x))
        end
    end
    @test all(pass)
end


end